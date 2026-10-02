"""Pretrained inference on project train; no training or evaluation."""
import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODELS = {'yolov8n': 'yolov8n.pt', 'lp_detection': 'LP-detection.pt'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def select_images(limit, seed):
    images = read_csv(ROOT/'data_processed/images.csv')
    splits = read_csv(ROOT/'data_processed/splits.csv')
    lookup = {r['image_path']: r for r in images}
    split_lookup = {r['image_path']: r for r in splits}
    if len(lookup)!=len(images) or len(split_lookup)!=len(splits) or lookup.keys()!=split_lookup.keys():
        raise ValueError('Inventory and split must contain the same unique image paths')
    group_splits = {}
    for row in splits:
        split = row['project_split']
        if split not in {'train','validation','test'} or not row['group_id']:
            raise ValueError('Invalid split or group')
        if group_splits.setdefault(row['group_id'],split)!=split:
            raise ValueError('A group crosses splits')
    selected = sorted([r for r in images if split_lookup[r['image_path']]['project_split']=='train'],key=lambda r:r['image_path'])
    if not selected:raise ValueError('No train images')
    if limit:
        buckets = {}
        for row in selected:buckets.setdefault(row['source'],[]).append(row)
        rng=random.Random(seed)
        for rows in buckets.values():rng.shuffle(rows)
        selected=[]
        while any(buckets.values()) and len(selected)<limit:
            for source in sorted(buckets):
                if buckets[source] and len(selected)<limit:selected.append(buckets[source].pop())
    for row in selected:
        path=(ROOT/row['image_path']).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('Invalid image path: '+str(path))
    return selected


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--limit',type=int,default=0,help='0 = all train; otherwise a source-balanced sample')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cpu',help='cpu, mps, or CUDA index such as 0')
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--iou',type=float,default=.7,help='NMS overlap threshold, NOT evaluation IoU')
    parser.add_argument('--examples',type=int,default=8)
    parser.add_argument('--output',type=Path,help='New output directory; never overwritten')
    args=parser.parse_args()
    if args.limit<0 or args.examples<0 or args.imgsz<=0 or not 0<=args.conf<=1 or not 0<=args.iou<=1:
        parser.error('Invalid counts, image size or thresholds')
    weights=HERE/'models'/MODELS[args.model]
    if not weights.is_file():raise FileNotFoundError(weights)
    selected=select_images(args.limit,args.seed)
    output=args.output or HERE/'results'/f'{args.model}_train'
    if output.exists():raise FileExistsError(f'{output} already exists; choose a new --output')
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    model=YOLO(str(weights),task='detect')
    if model.task!='detect':raise ValueError('Expected an object detection model')
    names={int(k):str(v) for k,v in model.names.items()}
    output.mkdir(parents=True,exist_ok=False)
    (output/'examples').mkdir()
    info=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),model=args.model,
              weights=str(weights.relative_to(ROOT)),weights_sha256=sha(weights),classes=names,
              project_split='train',selected_images=len(selected),source_counts=dict(Counter(r['source'] for r in selected)),
              sampling='source-balanced sample' if args.limit else 'all train',seed=args.seed,
              requested_device=args.device,imgsz=args.imgsz,confidence_threshold=args.conf,nms_iou=args.iou,
              max_det=300,agnostic_nms=False,coordinates='xyxy in original image pixels',
              versions=dict(python=platform.python_version(),ultralytics=ultralytics.__version__,torch=torch.__version__,opencv=cv2.__version__),
              input_sha256={name:sha(ROOT/'data_processed'/name) for name in ['images.csv','splits.csv']},
              note='Inference only. Confidence is not measured accuracy; no ground-truth metrics calculated.')
    def save_info(): (output/'run_info.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    save_info()
    try:
        total=0
        with (output/'predictions.csv').open('w',newline='') as pf,(output/'images_summary.csv').open('w',newline='') as sf:
            predictions=csv.DictWriter(pf,fieldnames=['image_path','source','project_split','detection_id','class_id','class_name','confidence','x1','y1','x2','y2'])
            summaries=csv.DictWriter(sf,fieldnames=['image_path','source','project_split','image_w','image_h','n_detections','class_counts','example_path'])
            predictions.writeheader();summaries.writeheader()
            for index,row in enumerate(selected,1):
                path=row['image_path']
                image=cv2.imread(str(ROOT/path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None:raise OSError('Unreadable image: '+path)
                h,w=image.shape[:2]
                if (w,h)!=(int(row['image_w']),int(row['image_h'])):raise ValueError('Unexpected dimensions: '+path)
                result=model.predict(source=image,device=args.device,imgsz=args.imgsz,conf=args.conf,iou=args.iou,max_det=300,agnostic_nms=False,verbose=False,save=False)[0]
                if tuple(result.orig_shape)!=(h,w):raise ValueError('Unexpected prediction coordinate space')
                boxes=result.boxes.xyxy.cpu().numpy();classes=result.boxes.cls.cpu().numpy().astype(int);scores=result.boxes.conf.cpu().numpy()
                for ident,(box,cls,score) in enumerate(zip(boxes,classes,scores)):
                    predictions.writerow(dict(image_path=path,source=row['source'],project_split='train',detection_id=ident,class_id=int(cls),class_name=names[int(cls)],confidence=float(score),**dict(zip(['x1','y1','x2','y2'],map(float,box)))))
                example=''
                if index<=args.examples:
                    example=f'examples/{index:04d}_{row["source"]}_{Path(path).stem}.jpg'
                    annotated=result.plot()
                    if annotated.shape[1]>1400:
                        annotated=cv2.resize(annotated,(1400,max(1,round(annotated.shape[0]*1400/annotated.shape[1]))),interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(output/example),annotated):raise OSError('Cannot save example')
                summaries.writerow(dict(image_path=path,source=row['source'],project_split='train',image_w=w,image_h=h,n_detections=len(boxes),class_counts=json.dumps(dict(Counter(names[int(c)] for c in classes))),example_path=example))
                total+=len(boxes)
                if index%50==0 or index==len(selected):print(f'{index}/{len(selected)} images; {total} detections',flush=True)
        info.update(status='complete',finished_at=datetime.now(timezone.utc).isoformat(),actual_device=str(model.device),detections=total)
        save_info()
    except Exception as exc:
        info.update(status='failed',error=str(exc));save_info();raise
    print('Classes:',names)
    print('Results:',output)


if __name__=='__main__':main()
