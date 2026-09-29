# Red Pill: Red Neuronal desde Cero con NumPy puro

> *"Tomas la pastilla azul… la historia termina. Tomas la roja… te enseño qué tan profundo llega la madriguera."* — Morpheus

Una **MLP completa implementada a mano**: capa densa, ReLU, softmax, cross-entropy y **backpropagation manual**.
Sin TensorFlow, sin Keras, sin PyTorch, sin autograd, sin `.fit()`. Solo `NumPy` y `Matplotlib`, y la regla de la cadena.

Clasifica dígitos MNIST escritos a mano con **~98% de accuracy en test**.

## ¿Qué hay dentro?

```
├── notebooks/
│   └── neural_numpy.ipynb             ← El notebook completo (comentado línea a línea, ya ejecutado)
├── data/mnist/                         ← Los 4 .gz de MNIST en formato IDX clásico
├── requirements.txt                    ← Dependencias
└── README.md
```

**Guía de estudio:** el notebook viene **comentado línea a línea** (borrá los comentarios a medida que los entiendas).

El notebook está estructurado así:

1. **Carga y exploración** — MNIST se descarga en formato IDX y se parsea *a mano* con `np.frombuffer`.
2. **Implementación** — `DenseLayer` (inicialización He, forward, backward), `ReLU`, `Softmax` y `CrossEntropyLoss` desde cero, cada uno con su visualización.
3. **Gradient check** — el backprop analítico se valida contra **gradientes numéricos** (diferencias finitas) antes de entrenar.
4. **Entrenamiento** — backprop manual, gradiente descendente por mini-batches, curvas de loss/accuracy por epoch,
   guardado automático del mejor modelo, early stopping y chequeo de neuronas muertas.
5. **Evaluación** — accuracy en test, predicciones correctas/incorrectas, matriz de confusión y análisis de errores.
6. **Conclusiones** — la celda obligatoria: qué se entiende del `.fit()` que antes no se entendía.

> Todo está dentro del notebook: no hay `src/` ni módulo externo que haya que instalar.

## Resultados obtenidos (un run de ejemplo)

| Métrica | Valor |
|---|---|
| Gradient check (error relativo máx) | `6.8e-07` ✅ |
| Mejor `val_loss` | `0.0706` (epoch 11) |
| Early stopping | cortó en el epoch 16 (5 epochs sin mejorar) |
| Accuracy en validación (del modelo elegido) | 98.16 % |
| **Accuracy en test** | **97.78 %** (222 errores de 10.000) |
| Peor clase / confusión más frecuente | dígito 5 (0.966) · `9 → 4` (14 casos) |

La arquitectura: `784 → 128 (ReLU) → 64 (ReLU) → 10 (logits)` con softmax + cross-entropy. Se entrena con
mini-batches de 128, `lr=0.2` —que se divide a la mitad en el epoch 10— e inicialización He, con un tope de 30
epochs que el early stopping nunca alcanza: corta en el 16.

### Sobre el overfitting y el "mejor modelo"

El `train_loss` sigue bajando (0.372 → 0.012) con el train en 99.80%, mientras el `val_loss` deja de bajar en el
epoch 11 y después oscila entre 0.071 y 0.077: la red está **memorizando** el train. Por eso `train()` hace dos cosas:

- **Guarda el mejor modelo**: cada epoch que bate el mínimo del `val_loss` por más de `min_delta = 0.001` saca una
  copia de los pesos (`results/best_model.npz`). Al terminar se **restauran esos pesos**, no los del último epoch.
- **Corta si deja de mejorar** (`patience=5`): seguir entrenando cuando la validación empeora solo quema cómputo.

El `min_delta` no es un detalle: entre los epochs 12 y 16 el `val_loss`rueda ±0.003 sin tendencia. Sin un umbral,
una bajada de 0.0001 —ruido puro— reiniciaría el contador y la red "parecería" seguir mejorando para siempre.

La selección se hace **con el set de validación, nunca con el de test**: el test se mira una sola vez, al final.
Y ojo con un detalle que confunde: el accuracy más alto en validación (98.22%) **no** es el del modelo evaluado,
porque cae en otro epoch distinto al de menor `val_loss`. El número que corresponde es el 98.16% del epoch 11.

Y un matiz que el notebook reconoce explícitamente: elegir el "mejor" epoch **también es una forma de sobreajustar**.
El 0.0706 del epoch 11 está casi dos desvíos estándar por debajo de sus vecinos (0.072–0.075), así que parte de esa
ventaja es ruido de medición. Lo correcto sería promediar los pesos de los mejores N epochs.

## Cómo correrlo

Requisitos: **Python 3.8+**, `numpy`, `matplotlib`, y `jupyter` / `nbconvert` para el notebook.

```bash
pip install -r requirements.txt
jupyter notebook notebooks/neural_numpy.ipynb
```

Los 4 `.gz` de MNIST (~11 MB) ya están en `data/mnist/`; la celda de descarga los vuelve a bajar solo si faltan.
Todo el código del modelo está *dentro del notebook*, sin dependencias adicionales.

Para re-ejecutar todo de punta a punta:

```bash
python -m nbconvert --to notebook --execute --inplace notebooks/neural_numpy.ipynb
```

---

## La pregunta del millón: ¿**por qué** backpropagation funciona?

Con mis palabras, sin Wikipedia.

Backpropagation no es un truco, es **simplemente aplicar la regla de la cadena del cálculo, pero al revés de como en el colegio**.

**La idea de fondo.** La red es una *composición* de funciones: la imagen entra, pasa por una multiplicación con pesos `W1`, una ReLU, otra multiplicación con `W2`, otra ReLU, otra multiplicación con `W3`, y termina como 10 logits. En el medio queda el piso de la montaña: la pérdida `L`, que en mi código es cross-entropy. Esa `L` es una función gigante y anidada de más de 100.000 parámetros, donde unos están "adentro" de otros.

**El único dato útil que queremos** de cada peso `w` es cuánto cambia la pérdida si `w` cambia un poquito, o sea `∂L/∂w`. Si tenemos eso para todos los pesos, tenemos la "dirección de máxima pendiente" de esa montaña gigante, y mover cada peso *en contra* de esa dirección hace que `L` baje: eso es todo el entrenamiento.

**El problema** es que calcular esa derivada para cada peso "desde cero" costaría muchísimo. Acá es donde la regla de la cadena hace magia: como la red es una composición `L(g(f(x)))`, entonces

```
∂L/∂w = (derivada de L respecto a lo que viene después) × (derivada de ese "después" respecto a w)
```

Es decir, la derivada de un peso de la **primera capa** depende de las derivadas de **todas** las capas que le siguen. Ese encadenamiento se puede calcular de una sola vez si lo hacemos **hacia atrás**, desde la salida hacia la entrada: la derivada de la última capa se calcula una vez, la capa anterior la reutiliza, la anterior a esa reutiliza la de esa, etc. Ningún cálculo se repite.

**Por qué funciona.** Porque una derivada "encadenada" no es más que *repartir la culpa proporcional a la influencia*. El error total le "avisa" a cada peso lo que le corresponde: los pesos que más contribuyeron al error reciben un gradiente más grande y se ajustan más; los que no influyeron, casi no cambian. Repitiendo esto miles de veces, la montaña se baja paso a paso y la red deja de equivocarse en lo que antes se equivocaba más. No hay inteligencia ni intención: hay una función que baja porque calculamos dónde está la pendiente y la seguimos en contra.

**Por qué parece mágico (y no lo es).** Lo que impresiona es que esa cadena de culpas se transmite por capas de transformaciones no lineales (las ReLUs) y, aun así, el gradiente llega a los pesos más profundos. Que llegue *bien* depende de cosas concretas que en este notebook se ven a mano:

- la inicialización (He) mantiene las magnitudes de los gradientes en un rango sano;
- la derivada `dL/dz = p − y` (softmax + cross-entropy combinados) es la puerta de entrada del error, y es trivial;
- la ReLU deja pasar el gradiente donde estaba "encendida" y lo corta donde estaba "apagada";
- los `X.T` y las transposiciones que rompen los shapes a las 2am no son parches: son la regla de la cadena pidiendo que las dimensiones cierren.

El momento en que el gradient check (backprop analítico vs. gradientes numéricos) coincide en `1e-7` es el momento
en que entendés que no hay magia: hay cálculo, y tu cuenta coincide con la de la computadora —que sacó la misma
derivada perturbando el peso y después voltándolo a su lugar—. La única diferencia entre hacer eso y hacer backprop
es la velocidad.
