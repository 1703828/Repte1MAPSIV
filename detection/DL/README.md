# Inferència YOLO preentrenada

Treballa amb `project_split=train` de `data_processed/splits.csv` i amb les dimensions d'`images.csv`. No entrena ni calcula mètriques contra ground truth. No necessita exportar etiquetes ni copiar les imatges. `plates.csv` s'utilitzarà en l'avaluació posterior.

Models del material docent, copiats sense canvis a `models/`: `yolov8n.pt` i `LP-detection.pt`. Les classes reals es consulten al model i es guarden a cada execució. No s'assumeix que la classe 0 signifiqui el mateix als dos models.

## Preparació

Entorn local al projecte (no cal pujar-lo a Git):

```sh
python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
python -m pip install -r detection/DL/requirements.txt
```

## Prova petita, des de l'arrel

```sh
python detection/DL/YOLO.py --model yolov8n --limit 10 --output detection/DL/results/yolov8n_train_sample
python detection/DL/YOLO.py --model lp_detection --limit 10 --output detection/DL/results/lp_detection_train_sample
```

El mateix límit i seed seleccionen les mateixes imatges per als dos models, alternant fonts. La mostra no representa proporcionalment tot train. Sense `--limit` es processen totes les imatges:

```sh
python detection/DL/YOLO.py --model yolov8n
python detection/DL/YOLO.py --model lp_detection
```

CPU per defecte. `--device mps` o `--device 0` permeten escollir acceleració si està disponible; no es pressuposa GPU. Paràmetres inicials: `--imgsz 640 --conf 0.25 --iou 0.7`. Aquí `iou` controla NMS, no el llindar de correcció contra GT. Les imatges d'OpenCV s'envien en BGR; les coordenades retornades són de la imatge original.

## Sortides

- `run_info.json`: estat complete/failed, classes, pesos i empremta, versions, dispositiu, paràmetres i entrades.
- `predictions.csv`: una fila per detecció amb classe, confiança i caixa xyxy en píxels originals.
- `images_summary.csv`: una fila per foto, també quan no es detecta res.
- `examples/`: fins a 8 imatges anotades (`--examples` per canviar-ho).

Cada execució exigeix una carpeta nova: utilitza `--output` si la ruta ja existeix. En cas d'error, la carpeta parcial es conserva i queda marcada com failed. Els fitxers de dos models es mantenen separats. La confiança no és una mesura d'exactitud; aquests resultats encara no són precisió/recall/mAP. No s'utilitzen validation ni test en aquesta fase.
