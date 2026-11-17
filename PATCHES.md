# Cambios sobre el código original y hallazgos

Todas las modificaciones están marcadas con `# PATCH` en el código.

## Correcciones para que el código corra (no cambian el método)

| Archivo | Problema en el original | Corrección |
| --- | --- | --- |
| `utils.py` | Importa `get_dataset` desde `data.py`, pero allí la función se llama `get_Code4ML`; además `get_dataset` se llamaba a sí misma | Se importa y se llama `get_Code4ML` |
| `demo.py` | `np.array(results)` mezcla números y una lista (`best_hp`): falla en numpy 1.24 o superior | Se calcula el mejor F2 con `max(...)` |
| `demo.py` | No tenía semilla ni forma de nombrar los resultados, aunque el README menciona `--seed` | Se agregan `--seed` y `--tag` |
| `data.py` | `pad_to_max_length` ya no existe en versiones recientes de transformers | `padding="max_length", truncation=True` (mismo largo: 512) |
| `nets.py` | `transformers.AdamW` fue eliminado | `torch.optim.AdamW` con `weight_decay=0.0` y `eps=1e-8`, igual que antes |
| `strategy.py` | API de OpenAI antigua (`openai.ChatCompletion`), clave escrita en el código | Cliente `openai>=1.0`, clave desde la variable `OPENAI_API_KEY`, creado solo cuando se necesita |
| `strategy.py` | Si aún no hay positivos etiquetados, `no_augmented_samples` queda sin definir y el programa se cae | Se inicializa en 0 |

## Configuración real encontrada en el código (el artículo no la reporta)

| Parámetro | Valor |
| --- | --- |
| Modelo | `microsoft/codebert-base` (`RobertaForSequenceClassification`, 2 clases) |
| Learning rate | 5e-5, scheduler lineal sin warmup |
| Batch size | 16 |
| Épocas | 4 (se reentrena desde cero en cada ronda de active learning) |
| Largo máximo | 512 tokens |
| Texto | Se pasa a minúsculas |
| Aumento de datos y prompting | `gpt-3.5-turbo-1106` |
| Aumento | Por cada positivo etiquetado se piden `round(negativos/positivos) - 1` reescrituras |
| Valores por defecto en `demo.py` | `n_init_labeled = 20`, `n_query = 20`, `patient = 5` (el artículo usa N = 50) |

## Observaciones al código original

1. **El criterio de parada del active learning usa el conjunto de test.** En `demo.py` se calcula el F2 sobre el test en cada ronda y se detiene cuando ese F2 no mejora en 5 rondas. El artículo afirma que el test no participa en ninguna decisión del modelo; en el código publicado sí participa en la parada. Qué ronda se reporta en la Tabla 3 no está indicado.
2. **Least confidence sobre logits.** `predict_prop_bert` devuelve logits, no probabilidades, y la consulta ordena por el logit máximo. La fórmula del artículo usa probabilidades.
3. **No hay código para BERT.** El código solo carga CodeBERT; la fila BERT de la Tabla 2 requeriría cambiar el modelo y el tokenizador.
4. **El prompting con pocos ejemplos está comentado.** `gpt.py` ejecuta solo zero-shot; los ejemplos de 1, 3 y 5 están comentados.
5. **Dataset:** train tiene 91 positivos y test 25 (total 116; el artículo indica 115).
