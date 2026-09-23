"""Group images for a future split, assuming every compatible UC3M pair matches.

Run: python3 scripts/review_vehicle_matches.py
Standard library only. Does not change images, annotations or inventory CSVs.
Outputs CSVs and a size report in vehicle_groups/.
Groups are conservative split units, not verified vehicle identities.
"""
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'vehicle_groups'


def compatible(a, b):
    return bool(a) and len(a) == len(b) and all(x == y or x == '*' or y == '*' for x, y in zip(a, b))


class UnionFind:
    def __init__(self, keys):
        self.parent = {k: k for k in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_groups(images, plates):
    image_by_path = {r['image_path']: r for r in images}
    if len(image_by_path) != len(images):
        raise ValueError('Duplicate image_path in images.csv')
    keys = [(r['image_path'], r['plate_id']) for r in plates]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate image_path / plate_id in plates.csv')
    for r in plates:
        if r['image_path'] not in image_by_path:
            raise ValueError('Annotation without image: ' + r['image_path'])
        if r['source'] != image_by_path[r['image_path']]['source']:
            raise ValueError('Conflicting source: ' + r['image_path'])
    # Image-level nodes automatically keep all plates in a photo together.
    groups = UnionFind(image_by_path)
    cv_vehicles = {}
    missing_cv = []
    for r in plates:
        if r['source'] == 'cv':
            vehicle = r['vehicle_id'].strip()
            if not vehicle:
                missing_cv.append(r['image_path'])
                continue
            previous = cv_vehicles.setdefault(vehicle, r['image_path'])
            groups.union(previous, r['image_path'])
    uc3m = sorted((r for r in plates if r['source'] == 'uc3m'), key=lambda r: (r['image_path'], r['plate_id']))
    candidates = []
    for i, a in enumerate(uc3m):
        for b in uc3m[i + 1:]:
            if a['image_path'] == b['image_path'] or not compatible(a['plate_text'], b['plate_text']):
                continue
            groups.union(a['image_path'], b['image_path'])
            candidates.append(dict(image_a=a['image_path'], plate_id_a=a['plate_id'],
                                   image_b=b['image_path'], plate_id_b=b['plate_id'],
                                   decision='same', basis='assumed_compatible_text',
                                   cross_original_split=image_by_path[a['image_path']]['split'] != image_by_path[b['image_path']]['split']))
    members = defaultdict(list)
    for path in sorted(image_by_path):
        members[groups.find(path)].append(path)
    assignments, summaries = [], []
    plate_counts = Counter(r['image_path'] for r in plates)
    for paths in members.values():
        group_id = 'g_' + hashlib.sha256('\n'.join(paths).encode()).hexdigest()[:16]
        sources = sorted({image_by_path[p]['source'] for p in paths})
        original = Counter(image_by_path[p]['split'] for p in paths)
        summaries.append(dict(group_id=group_id, source=';'.join(sources), n_images=len(paths),
                              n_plates=sum(plate_counts[p] for p in paths),
                              original_train=original['train'], original_test=original['test']))
        for path in paths:
            r = image_by_path[path]
            assignments.append(dict(image_path=path, source=r['source'], original_split=r['split'], group_id=group_id))
    assignments.sort(key=lambda r: r['image_path'])
    summaries.sort(key=lambda r: (-r['n_images'], r['group_id']))
    return assignments, summaries, candidates, missing_cv


def main():
    images = read_csv(ROOT / 'data_processed/images.csv')
    plates = read_csv(ROOT / 'data_processed/plates.csv')
    assignments, summaries, candidates, missing_cv = build_groups(images, plates)
    lookup = {r['image_path']: r['group_id'] for r in assignments}
    assert len(lookup) == len(images)
    assert all(lookup[p['image_a']] == lookup[p['image_b']] for p in candidates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / 'image_groups.csv', ['image_path', 'source', 'original_split', 'group_id'], assignments)
    write_csv(OUTPUT / 'group_sizes.csv', ['group_id', 'source', 'n_images', 'n_plates', 'original_train', 'original_test'], summaries)
    write_csv(OUTPUT / 'assumed_matches.csv', ['image_a', 'plate_id_a', 'image_b', 'plate_id_b', 'decision', 'basis', 'cross_original_split'], candidates)
    lines = ['# Agrupación automática para la partición', '',
             'Todas las parejas candidatas de UC3M se asumen como el mismo vehículo, por decisión del proyecto. No son coincidencias verificadas manualmente.', '',
             'Se comparan textos de igual longitud: cada posición debe coincidir o contener un asterisco en alguno de los textos. Se unen las coincidencias en cadena y las imágenes con matrículas compartidas. CV se agrupa por vehicle_id. No se buscan coincidencias entre CV y UC3M.', '',
             'Un group_id representa imágenes que deben permanecer juntas en la partición; puede incluir varios vehículos. Las imágenes sin coincidencias también reciben grupo. Los errores de transcripción pueden impedir detectar repeticiones.', '',
             f'- Imágenes: {len(images)}', f'- Grupos: {len(summaries)}',
             f'- Parejas UC3M asumidas como coincidencias: {len(candidates)}',
             f'- Parejas entre train/test originales: {sum(p["cross_original_split"] for p in candidates)}',
             f'- Anotaciones CV sin vehicle_id: {len(missing_cv)}', '',
             '| Fuente | Imágenes | Grupos | Grupo mayor (imágenes) |', '|---|---:|---:|---:|']
    for source in sorted({r['source'] for r in images}):
        ss = [r for r in summaries if r['source'] == source]
        lines.append(f'| {source} | {sum(r["source"] == source for r in images)} | {len(ss)} | {max((r["n_images"] for r in ss), default=0)} |')
    lines += ['', '## Diez grupos más grandes', '', '| Grupo | Fuente | Imágenes | Matrículas |', '|---|---|---:|---:|']
    lines += [f'| {r["group_id"]} | {r["source"]} | {r["n_images"]} | {r["n_plates"]} |' for r in summaries[:10]]
    lines += ['', '## Archivos', '',
              '- image_groups.csv: una fila por imagen, con grupo y partición original.',
              '- group_sizes.csv: tamaños de todos los grupos.',
              '- assumed_matches.csv: parejas candidatas con decision=same y basis=assumed_compatible_text.', '',
              'No se han creado train/validation/test nuevos ni modificado los datos originales. Para regenerar, ejecuta `python3 scripts/review_vehicle_matches.py`; sobrescribe únicamente estos informes automáticos. No requiere Pillow ni revisión HTML.', '']
    (OUTPUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines[6:13]))
    print(f'Report: {OUTPUT / "report.md"}')


if __name__ == '__main__':
    main()
