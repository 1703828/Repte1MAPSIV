# Agrupación automática para la partición

Todas las parejas candidatas de UC3M se asumen como el mismo vehículo, por decisión del proyecto. No son coincidencias verificadas manualmente.

Se comparan textos de igual longitud: cada posición debe coincidir o contener un asterisco en alguno de los textos. Se unen las coincidencias en cadena y las imágenes con matrículas compartidas. CV se agrupa por vehicle_id. No se buscan coincidencias entre CV y UC3M.

Un group_id representa imágenes que deben permanecer juntas en la partición; puede incluir varios vehículos. Las imágenes sin coincidencias también reciben grupo. Los errores de transcripción pueden impedir detectar repeticiones.

- Imágenes: 2073
- Grupos: 1109
- Parejas UC3M asumidas como coincidencias: 1398
- Parejas entre train/test originales: 446
- Anotaciones CV sin vehicle_id: 0

| Fuente | Imágenes | Grupos | Grupo mayor (imágenes) |
|---|---:|---:|---:|
| cv | 98 | 63 | 5 |
| uc3m | 1975 | 1046 | 406 |

## Diez grupos más grandes

| Grupo | Fuente | Imágenes | Matrículas |
|---|---|---:|---:|
| g_f4cfdc3c3900d3c2 | uc3m | 406 | 689 |
| g_2d2ad90e5c5e63aa | uc3m | 36 | 57 |
| g_83ae8f5ba117fee3 | uc3m | 18 | 36 |
| g_f63e6c473156bdf4 | uc3m | 14 | 21 |
| g_a36330f1784654ef | uc3m | 12 | 25 |
| g_b9aa2582ead1fb65 | uc3m | 12 | 16 |
| g_1580c0403da19b96 | uc3m | 11 | 16 |
| g_f8086cf86b742658 | uc3m | 11 | 16 |
| g_9ac94bd3810a499f | uc3m | 10 | 11 |
| g_230e2a4e8128540f | uc3m | 9 | 15 |

## Archivos

- image_groups.csv: una fila por imagen, con grupo y partición original.
- group_sizes.csv: tamaños de todos los grupos.
- assumed_matches.csv: parejas candidatas con decision=same y basis=assumed_compatible_text.

No se han creado train/validation/test nuevos ni modificado los datos originales. Para regenerar, ejecuta `python3 scripts/review_vehicle_matches.py`; sobrescribe únicamente estos informes automáticos. No requiere Pillow ni revisión HTML.
