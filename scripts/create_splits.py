"""Reproducible grouped 70/15/15 image split; standard library only.

Run: python3 scripts/create_splits.py
Writes data_processed/splits.csv and vehicle_groups/split_report.md.
Original images, annotations and original UC3M split remain unchanged.
"""
import csv
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SPLITS = ('train', 'validation', 'test')
RATIOS = (.70, .15, .15)
INPUTS = ('data_processed/images.csv', 'data_processed/plates.csv', 'vehicle_groups/image_groups.csv')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def unique_by_path(rows, name):
    result = {r['image_path']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate image_path in {name}')
    return result


def build_features(images, plates, mapping):
    image_map = unique_by_path(images, 'images.csv')
    group_map = unique_by_path(mapping, 'image_groups.csv')
    if image_map.keys() != group_map.keys():
        raise ValueError('Image inventory and grouping have different image paths. Regenerate groups first.')
    annotations = defaultdict(list)
    seen = set()
    for row in plates:
        key = (row['image_path'], row['plate_id'])
        if key in seen or row['image_path'] not in image_map:
            raise ValueError('Duplicate annotation or unknown image: ' + str(key))
        if row['source'] != image_map[row['image_path']]['source']:
            raise ValueError('Conflicting annotation source')
        seen.add(key)
        annotations[row['image_path']].append(row)
    groups = defaultdict(Counter)
    image_features = {}
    for path, row in sorted(image_map.items()):
        g = group_map[path]
        if not g['group_id'] or g['source'] != row['source'] or g['original_split'] != row['split']:
            raise ValueError('Invalid or stale group metadata: ' + path)
        source = row['source']
        f = Counter({'images': 1, 'source:' + source: 1})
        f['view:' + source + ':' + (row['view'] or 'unknown')] = 1
        # Count images having a condition, not the number of plate instances.
        lighting = {r['lighting'].strip() for r in annotations[path] if r['lighting'].strip()}
        lighting = {'day' if x == 'daylight' else x for x in lighting} or {'unknown'}
        for value in lighting:
            f['lighting:' + source + ':' + value] = 1
        shapes = {r['plate_shape'] for r in annotations[path] if r['plate_shape']} or {'unknown'}
        for value in shapes:
            f['shape:' + source + ':' + value] = 1
        image_features[path] = f
        groups[g['group_id']].update(f)
    return dict(groups), image_features


def allocate(groups, seed=SEED):
    if len(groups) < 3:
        raise ValueError('At least three groups are required')
    keys = sorted({k for f in groups.values() for k in f})
    vectors = {g: [f[k] for k in keys] for g, f in groups.items()}
    totals = [sum(v[j] for v in vectors.values()) for j in range(len(keys))]
    weights = [8 if k == 'images' else 4 if k.startswith('source:') else 2 if k.startswith('view:') else 1 if k.startswith('lighting:') else .5 for k in keys]
    scale = [w / max(1, t) for w, t in zip(weights, totals)]
    targets = [[ratio * t for t in totals] for ratio in RATIOS]
    largest = min(groups, key=lambda g: (-groups[g]['images'], g))
    best = None
    for attempt in range(12):
        rng = random.Random(seed + attempt)
        counts = [[0] * len(keys) for _ in SPLITS]
        assigned = {largest: 0}
        counts[0] = vectors[largest].copy()
        rest = sorted(g for g in groups if g != largest)
        rng.shuffle(rest)
        rest.sort(key=lambda g: -groups[g]['images'])

        def delta(g, split, sign):
            return sum(s * (2 * (c - t) * sign * v + v * v)
                       for s, c, t, v in zip(scale, counts[split], targets[split], vectors[g]))

        def add(g, split, sign):
            counts[split] = [c + sign * v for c, v in zip(counts[split], vectors[g])]

        for g in rest:
            options = list(range(3))
            rng.shuffle(options)
            chosen = min(options, key=lambda split: delta(g, split, 1))
            assigned[g] = chosen
            add(g, chosen, 1)
        # Improve the balance through whole-group moves. Never move the largest.
        for _ in range(30):
            changed = False
            rng.shuffle(rest)
            for g in rest:
                old = assigned[g]
                removal = delta(g, old, -1)
                alternatives = [(removal + delta(g, new, 1), new) for new in range(3) if new != old]
                gain, new = min(alternatives)
                if gain < -1e-10:
                    add(g, old, -1)
                    add(g, new, 1)
                    assigned[g] = new
                    changed = True
            if not changed:
                break
        score = sum(s * (c-t)**2 for count, target in zip(counts, targets) for s, c, t in zip(scale, count, target))
        if best is None or score < best[0]:
            best = score, assigned.copy()
    result = {g: SPLITS[n] for g, n in best[1].items()}
    if set(result.values()) != set(SPLITS):
        raise ValueError('Optimization produced an empty split')
    return result, largest, best[0]


def main():
    images, plates, mapping = [read_csv(ROOT / p) for p in INPUTS]
    groups, image_features = build_features(images, plates, mapping)
    assignment, largest, score = allocate(groups)
    rows = [dict(image_path=r['image_path'], group_id=r['group_id'], project_split=assignment[r['group_id']]) for r in sorted(mapping, key=lambda r: r['image_path'])]
    by_path = unique_by_path(rows, 'splits.csv')
    if set(by_path) != {r['image_path'] for r in images}:
        raise ValueError('Split does not cover all images')
    # Independent checks of group integrity and the original CV vehicle IDs.
    for field_rows, key in [(rows, 'group_id'), ([r for r in plates if r['source']=='cv' and r['vehicle_id']], 'vehicle_id')]:
        seen = defaultdict(set)
        for r in field_rows:
            seen[r[key]].add(by_path[r['image_path']]['project_split'])
        if any(len(s) != 1 for s in seen.values()):
            raise ValueError('Group/vehicle crosses splits')
    assumed = ROOT / 'vehicle_groups/assumed_matches.csv'
    if not assumed.exists():
        raise ValueError('Missing assumed_matches.csv; regenerate grouping first')
    for pair in read_csv(assumed):
        if by_path[pair['image_a']]['group_id'] != by_path[pair['image_b']]['group_id']:
            raise ValueError('An assumed pair crosses groups; regenerate grouping first')
    distribution = {s: Counter() for s in SPLITS}
    for path, f in image_features.items():
        distribution[by_path[path]['project_split']].update(f)
    group_counts = Counter(assignment.values())
    plate_counts = Counter(by_path[r['image_path']]['project_split'] for r in plates)
    lines = ['# Partición por grupos', '',
             f'Semilla: {SEED}. Objetivo por imágenes: 70 % train / 15 % validation / 15 % test.',
             'Se mantienen grupos completos. Se prueban 12 repartos reproducibles y se mejora su equilibrio moviendo grupos completos. Se minimiza el error cuadrático respecto a las proporciones objetivo, normalizado por el total de cada categoría: peso 8 para imágenes, 4 para fuente, 2 para vista, 1 para iluminación y 0,5 para forma de matrícula. No se utilizan resultados de los detectores.', '',
             f'El grupo mayor, {largest}, contiene {groups[largest]["images"]} imágenes y se fija en train: supera el tamaño objetivo de validation y test.', '',
             '| Conjunto | Imágenes | Porcentaje | Grupos | Matrículas |', '|---|---:|---:|---:|---:|']
    for s in SPLITS:
        n = distribution[s]['images']
        lines.append(f'| {s} | {n} | {100*n/len(images):.2f} % | {group_counts[s]} | {plate_counts[s]} |')
    lines += ['', '## Distribución por condiciones', '',
              'Cada celda cuenta imágenes. Una foto puede tener varias condiciones de iluminación o formas de matrícula: esas categorías pueden solaparse. unknown indica que no hay información. daylight se normaliza a day. No se inventan vistas para UC3M ni formas para CV.', '',
              '| Categoría | Train | Validation | Test | Total |', '|---|---:|---:|---:|---:|']
    for key in sorted({k for f in image_features.values() for k in f} - {'images'}):
        ns = [distribution[s][key] for s in SPLITS]
        lines.append('| ' + key + ' | ' + ' | '.join(map(str, ns + [sum(ns)])) + ' |')
    lines += ['', '## Verificaciones y límites', '',
              '- Todas las imágenes aparecen exactamente una vez; ningún grupo cruza conjuntos.',
              '- Los vehicle_id de CV y todas las parejas asumidas de UC3M permanecen juntos.',
              '- La partición propia reemplaza el uso del split original de UC3M en los experimentos. Las carpetas originales no cambian.',
              '- Las coincidencias de UC3M son una suposición conservadora, no identidades verificadas; no se comprobaron coincidencias entre CV y UC3M. No se garantiza detectar repeticiones con errores de anotación.',
              '- Los grupos grandes condicionan el equilibrio y pueden conectar vehículos diferentes. Los porcentajes son aproximados.',
              '- Usar esta misma partición para ML y YOLO; aumentos de entrenamiento solo desde train. Mantener test reservado para la evaluación final.',
              '- Una vez iniciados los experimentos, conservar splits.csv. Si cambian los datos o los grupos, revisar explícitamente si se debe crear una nueva versión de la partición.', '',
              '## Huellas de las entradas', '']
    for path in (*INPUTS, 'vehicle_groups/assumed_matches.csv'):
        lines.append(f'- `{path}`: `{hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}`')
    output = ROOT / 'data_processed/splits.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image_path', 'group_id', 'project_split'])
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / 'vehicle_groups/split_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    for s in SPLITS:
        print(s, distribution[s]['images'], 'images;', group_counts[s], 'groups')
    print(output)


if __name__ == '__main__':
    main()
