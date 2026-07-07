# Análisis de Resistencia a Compresión del Concreto
## 1. Dataset
**Nombre:** Concrete Compressive Strength Data Set
**Fuente:** UCI Machine Learning Repository / Kaggle
**Enlace:** https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set
Este dataset contiene 1030 registros de mezclas de concreto probadas en laboratorio. 
Cada fila representa una mezcla distinta, con 8 variables de entrada (ingredientes y edad) 
y 1 variable de salida (la resistencia a compresión resultante, medida en MPa).
**Variables del dataset:**
- Cemento (kg/m³)
- Escoria de alto horno (kg/m³)
- Cenizas volantes (kg/m³)
- Agua (kg/m³)
- Superplastificante (kg/m³)
- Agregado grueso (kg/m³)
- Agregado fino (kg/m³)
- Edad del concreto (días)
- Resistencia a compresión (MPa) — variable objetivo
## 2. Análisis de dominio
La resistencia a compresión del concreto es una de las propiedades mecánicas más 
importantes en ingeniería estructural, ya que determina la capacidad de un elemento 
(columna, viga, losa, etc.) para soportar cargas sin fallar. Es un valor fundamental 
en el diseño de cualquier estructura de concreto armado, desde edificios hasta puentes.
A diferencia de otros materiales de construcción, el concreto no tiene una resistencia 
fija: depende de la proporción de sus ingredientes (cemento, agua, agregados y aditivos) 
y del tiempo de curado (edad). Tradicionalmente, esta resistencia se estima mediante 
fórmulas empíricas o pruebas físicas de laboratorio (como el ensayo de compresión en 
cilindros de concreto a los 7, 14 o 28 días).
Sin embargo, la relación entre los ingredientes y la resistencia final **no es lineal**: 
por ejemplo, aumentar el agua reduce la resistencia (relación agua-cemento), pero 
aumentar el cemento no siempre la mejora de forma proporcional. Esta complejidad hace 
que el análisis de datos y las técnicas de aprendizaje automático sean útiles para 
predecir la resistencia sin necesidad de esperar los tiempos de curado tradicionales 
(hasta 28 días), lo cual representa un ahorro de tiempo y costos en obra.
Este dataset ha sido ampliamente utilizado en investigaciones de ingeniería civil para 
modelar y predecir la resistencia del concreto usando redes neuronales y otros modelos 
estadísticos, sentando un precedente en el uso de ciencia de datos aplicada a materiales 
de construcción.
## 3. Referencias
1. Yeh, I-C. (1998). *Modeling of strength of high performance concrete using 
   artificial neural networks*. Cement and Concrete Research, 28(12), 1797-1808.
2. Yeh, I-C. (2006). *Analysis of strength of concrete using design of experiments 
   and neural networks*. Journal of Materials in Civil Engineering, ASCE, 18(4), 597-604.
