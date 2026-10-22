# Predicción in-game del resultado de partidos del Mundial 2026

Proyecto de INF-396 (Ciencia de Datos, USM) Catalina Yañez y María-Belén Segura

Estimamos la probabilidad de victoria, empate o derrota al minuto 90 de un partido del Mundial 2026 a partir de su estado en los minutos 15, 30, 45, 60 y 75: marcador, xG, tiros, tiros al arco y contexto previo (Elo, ranking FIFA, valor del plantel). Compararemos Regresión Logística, SVM y KNN.

## 1. Preparar el entorno

```bash
git clone https://github.com/Clubbeb/Proyecto-Datos
cd Proyecto-Datos
python3 -m venv .venv
source .venv/bin/activate        # en Windows (PowerShell): .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Descargar los datos

Los datos **no están en el repositorio** (la carpeta `data/` está en `.gitignore`). Descárgalos de Kaggle con el botón *Download*:

| Carpeta | Dataset | Contenido |
| --- | --- | --- |
| `data/d1/` | [FIFA World Cup 2026 – Overview, Results & Stats](https://www.kaggle.com/datasets/binyoh/fifa-world-cup-2026-overview-results-and-stats) (≈30 MB) | Un JSON por partido desde la API de FotMob, con cada tiro, su minuto y su xG |
| `data/d2/` | [FIFA World Cup 2026 Dataset](https://www.kaggle.com/datasets/mominullptr/fifa-world-cup-2026-dataset) (≈340 KB) | CSV con partidos, marcadores, eventos y contexto previo (Elo, ranking, valor del plantel) |

Descomprímelos así (cambia la ruta por la de tus descargas):

```bash
mkdir -p data/d1 data/d2
unzip RUTA/AL/ZIP_DE_D1.zip -d data/d1
unzip RUTA/AL/ZIP_DE_D2.zip -d data/d2
```

Verifica que quedaron bien:

- `ls data/d1` debe mostrar `comp  matches  players  teams`, y `data/d1/matches/` debe tener 104 archivos `match_data_*.json`.
- `ls data/d2` debe mostrar `matches.csv`, `match_events.csv` y `match_prediction_features.csv`, entre otros.

## 3. Ejecutar los notebooks en orden

Cada notebook guarda su resultado en `data/processed/` y el siguiente lo lee. Ábrelos y usa **Run All**.

| Notebook | Qué hace | Genera |
| --- | --- | --- |
| `notebooks/01_exploracion.ipynb` | Primera exploración de D2 | — |
| `notebooks/02_tiros_fotmob.ipynb` | Lee los 104 JSON de FotMob; calcula tiros, tiros al arco, xG y goles acumulados por checkpoint; arma la etiqueta | `estado_d1.csv` |
| `notebooks/03_union_y_dataset.ipynb` | Empareja los partidos de D1 y D2; agrega el contexto previo; calcula las diferencias A − B | `mapa_d1_d2.csv`, `instances.csv` |

**Ojo con el orden si partes desde cero:** la celda del notebook 02 que verifica los autogoles lee `mapa_d1_d2.csv`, que se crea en el 03. Corre primero en el 03 solo las celdas que emparejan los partidos y guardan el mapa; después corre el 02 completo, y finalmente el 03 completo. (Esto se ordenará al pasar el código a scripts.)

**Resultado:** `data/processed/instances.csv`, con **520 instancias** (104 partidos × 5 checkpoints) y sin valores faltantes.

## 4. Convenciones del proyecto

- **Equipo A** = el listado como local en el fixture; **equipo B** = el visitante. Todas las variables de equipo son diferencias **A − B**.
- **Etiqueta `y`:** resultado al minuto 90 → `0` = gana A, `1` = empate, `2` = gana B. Los partidos con prórroga o penales cuentan como empate.
- **Checkpoints:** 15, 30, 45, 60 y 75. El 45 incluye el tiempo añadido del primer tiempo. La prórroga y la tanda de penales nunca se usan.
- **No usar** las estadísticas de `content.stats` (D1) ni de `match_team_stats.csv` (D2): son totales del partido completo y filtrarían el resultado.

## 5. Verificaciones hechas (para el informe)

| Verificación | Resultado |
| --- | --- |
| Partidos con tiros en FotMob | 104 de 104 |
| Tiros sin xG | 14, y son exactamente los autogoles |
| Goles sin la tanda de penales | 308 en D1, igual que en D2 (294 goles + 14 autogoles) |
| Autogoles | Ambos datasets los registran con el equipo que lo comete; sumándolos al rival, el marcador coincide con el oficial en 104 de 104 partidos |
| Nombres de selecciones | 6 escritos distinto en D1 y D2 (p. ej. *Ivory Coast* / *Côte d'Ivoire*); se unificaron con un diccionario |
| Fechas | Difieren en ±1 día en 42 de 104 partidos, en ambas direcciones; se emparejaron por selecciones con tolerancia de un día (correspondencia 1 a 1) |
| Local invertido entre datasets | 0 partidos |
| Distribución de la etiqueta | 46 victorias de A, 29 empates, 29 victorias de B |

