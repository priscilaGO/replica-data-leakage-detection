# Réplica: detección de data leakage en código de ML

Réplica del artículo:

> Alturayeif, N. & Hassine, J. (2025). *Data leakage detection in machine learning code: transfer learning, active learning, or low-shot prompting?* PeerJ Computer Science, 11, e2730. https://doi.org/10.7717/peerj-cs.2730

Trabajo del curso Propuesta de Investigación (Ingeniería de Sistemas, Universidad de Lima, 2026-2). Autora de la réplica: Priscila Meel Gallegos Ore.

## Qué se replica

| Enfoque | Resultado del paper (F2, promedio de 5 corridas) | Dónde |
| --- | --- | --- |
| CodeBERT afinado (transfer learning) | 0.67 | `AL4Code/demo.py` sin `--active_learning` |
| CodeBERT + active learning (least confidence) | 0.72 | `AL4Code/demo.py --active_learning --with_augmentation` |
| GPT con 0, 1, 3 y 5 ejemplos | 0.18 / 0.30 / 0.39 / 0.31 | `GPT4Code/gpt.py` |

## Estructura

- `Dataset/`: train (1,523 fragmentos) y test (381) publicados por los autores en Figshare (DOI 10.6084/m9.figshare.24893799.v1).
- `AL4Code/`: código de los autores (Figshare, DOI 10.6084/m9.figshare.24893832.v1), con correcciones mínimas de compatibilidad. Ver `PATCHES.md`.
- `GPT4Code/`: script de prompting de los autores (sin modificar por ahora).
- `Results/`: salidas de cada corrida.
- `notebooks/`: notebook de Colab para ejecutar la réplica paso a paso.

## Cómo ejecutar

Abrir `notebooks/01_codebert_pasivo.ipynb` en Google Colab con GPU (T4) y ejecutar las celdas en orden.

## Licencias

El código base de active learning viene de DeepAL4NLP (MIT, ver `LICENSE_AL4Code_MIT.txt`). Los datos y el código originales son de los autores del artículo; revisar la licencia indicada en Figshare antes de reutilizarlos.
