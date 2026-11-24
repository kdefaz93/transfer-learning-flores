# Clasificación de flores con Transfer Learning (Xception)

Clasificador de imágenes de **5 tipos de flores** construido con **aprendizaje por transferencia**: se reutiliza Xception, una red convolucional preentrenada en ImageNet, y se entrena un clasificador propio encima. Proyecto desarrollado con TensorFlow y Keras.

> **English summary:** Flower image classifier (5 classes) using transfer learning with a frozen Xception base (ImageNet) and a custom Dense/Dropout head. Test accuracy: **74.11%** with 71×71 inputs, trained on CPU.

## Resultados

| Conjunto | Precisión |
|---|---|
| Validación (mejor época) | 75.2% |
| **Prueba** | **74.11%** |

- `EarlyStopping` detuvo el entrenamiento en la época 9 y restauró los pesos de la época 6.
- La precisión en prueba es muy cercana a la de validación, por lo que el modelo no está sobreajustado a la validación.
- La precisión de entrenamiento es igual o menor que la de validación: el modelo se queda corto (subajuste), no memoriza.

<!-- Cuando guardes la gráfica en imagenes/, descomenta la línea siguiente:
![Curva de precisión](imagenes/curva_precision.png)
-->

## Datos

- **Dataset:** [`tf_flowers`](https://www.tensorflow.org/datasets/catalog/tf_flowers) (TensorFlow Datasets), 3,670 imágenes en 5 clases.
- **División:** 80% entrenamiento (2,936) · 10% validación (367) · 10% prueba (367), con semilla fija (`shuffle_seed=23`).
- **Preprocesamiento:** redimensionado a 71×71 y escalado de los píxeles al rango [-1, 1], como requiere Xception.

## Arquitectura

```text
Entrada (71×71×3)
   ↓
Xception (ImageNet, congelado, include_top=False)
   ↓
Flatten
   ↓
Dense(128, ReLU)
   ↓
Dropout(0.5)
   ↓
Dense(5, softmax)
```

- **Optimizador:** Adam
- **Pérdida:** `sparse_categorical_crossentropy`
- **Métrica:** `sparse_categorical_accuracy`
- **Callback:** `EarlyStopping` (monitor `val_loss`, `patience=3`, `restore_best_weights=True`)

## Cómo ejecutarlo

Requiere **Python 3.11**. Graphviz es opcional (solo para dibujar el diagrama del modelo con `plot_model`).

```bash
python -m venv venv311
venv311\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook aprendizaje_por_transferencia.ipynb
```

Nota: en Windows nativo TensorFlow entrena en CPU, por lo que cada época tarda alrededor de 20 a 25 segundos con este tamaño de imagen.

## Próximos pasos

- Aumentar la resolución de entrada (por ejemplo, 150×150).
- Sustituir `Flatten` por `GlobalAveragePooling2D` para reducir parámetros.
- Reducir el Dropout (por ejemplo, a 0.3).
- Aplicar fine-tuning: descongelar las últimas capas de Xception con una tasa de aprendizaje baja.
- Añadir aumento de datos (giros y rotaciones).

## Créditos

Aprendizaje basado en el módulo de Deep Learning del **IBM Machine Learning Professional Certificate** (Coursera) y en el canal de YouTube **Codificando Bits**.
