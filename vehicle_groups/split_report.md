# Partición por grupos

Semilla: 42. Objetivo por imágenes: 70 % train / 15 % validation / 15 % test.
Se mantienen grupos completos. Se prueban 12 repartos reproducibles y se mejora su equilibrio moviendo grupos completos. Se minimiza el error cuadrático respecto a las proporciones objetivo, normalizado por el total de cada categoría: peso 8 para imágenes, 4 para fuente, 2 para vista, 1 para iluminación y 0,5 para forma de matrícula. No se utilizan resultados de los detectores.

El grupo mayor, g_f4cfdc3c3900d3c2, contiene 406 imágenes y se fija en train: supera el tamaño objetivo de validation y test.

| Conjunto | Imágenes | Porcentaje | Grupos | Matrículas |
|---|---:|---:|---:|---:|
| train | 1451 | 70.00 % | 505 | 1976 |
| validation | 311 | 15.00 % | 302 | 337 |
| test | 311 | 15.00 % | 302 | 332 |

## Distribución por condiciones

Cada celda cuenta imágenes. Una foto puede tener varias condiciones de iluminación o formas de matrícula: esas categorías pueden solaparse. unknown indica que no hay información. daylight se normaliza a day. No se inventan vistas para UC3M ni formas para CV.

| Categoría | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| lighting:cv:artificial | 2 | 0 | 0 | 2 |
| lighting:cv:day | 66 | 15 | 15 | 96 |
| lighting:uc3m:day | 1175 | 251 | 252 | 1678 |
| lighting:uc3m:night | 210 | 45 | 45 | 300 |
| shape:cv:unknown | 68 | 15 | 15 | 98 |
| shape:uc3m:single_row | 1364 | 294 | 291 | 1949 |
| shape:uc3m:two_row | 21 | 4 | 5 | 30 |
| source:cv | 68 | 15 | 15 | 98 |
| source:uc3m | 1383 | 296 | 296 | 1975 |
| view:cv:frontal | 22 | 5 | 5 | 32 |
| view:cv:lateral | 46 | 10 | 10 | 66 |
| view:uc3m:unknown | 1383 | 296 | 296 | 1975 |

## Verificaciones y límites

- Todas las imágenes aparecen exactamente una vez; ningún grupo cruza conjuntos.
- Los vehicle_id de CV y todas las parejas asumidas de UC3M permanecen juntos.
- La partición propia reemplaza el uso del split original de UC3M en los experimentos. Las carpetas originales no cambian.
- Las coincidencias de UC3M son una suposición conservadora, no identidades verificadas; no se comprobaron coincidencias entre CV y UC3M. No se garantiza detectar repeticiones con errores de anotación.
- Los grupos grandes condicionan el equilibrio y pueden conectar vehículos diferentes. Los porcentajes son aproximados.
- Usar esta misma partición para ML y YOLO; aumentos de entrenamiento solo desde train. Mantener test reservado para la evaluación final.
- Una vez iniciados los experimentos, conservar splits.csv. Si cambian los datos o los grupos, revisar explícitamente si se debe crear una nueva versión de la partición.

## Huellas de las entradas

- `data_processed/images.csv`: `4aa1130e8d3528f707889b469eab5424f0ead5d70bb9b6b81e7e7bacde98eb25`
- `data_processed/plates.csv`: `befd5664b0cad1c8fdcc170809c0198f87a0bbf36f36b13cb8257f4c145643b3`
- `vehicle_groups/image_groups.csv`: `92b339f34701c09f566a87a0f5ea483c4e44c259d790a9458dd19e29999e78a2`
- `vehicle_groups/assumed_matches.csv`: `e7e6669e9a612ed25d86e2eb58e6a3b494aaf4e0cd92a7b859e99e47b73687ce`
