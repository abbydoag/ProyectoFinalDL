# Proyecto Final Deep Learning y Sistemas Inteligentes
## ⛽ ConsuGas: Sistema de Monitoreo de Combustible
**¿Qué es ConsuGas**
Es una aplicación que permite al usuario predecir el rendimiento de combustible de un vehículo (en millas o km/galón) y permite a los usuarios auditar su consumo real comparándolo con las predicciones del modelo de Deep Learning.

---

## Características principales

*   **Predicción con Deep Learning:** Utiliza una Red Neuronal Densa (MLP) entrenada con un dataset de vehículos modernos para estimar el rendimiento de combustible.
*   **Auditoría en tiempo real:** El usuario puede registrar sus visitas a la gasolinera (Trip, precio del galón y monto pagado) para calcular su consumo real y compararlo con la predicción del modelo.
*   **Elección de unidades de medición:** Soporte completo para millas (MPG) y kilómetros (km/galón).
*   **Cálculo de galones:** No es necesario calcular cuántos galones se compraron, la aplicación lo deduce dividiendo el monto pagado entre el precio del galón.
*   **Análisis:** Generación de gráficas dinámicas que comparan los galones esperados por el modelo contra los galones reales consumidos.
*   **Contexto local:** La interfaz hace uso de la moneda de Guatemala, quetzales.

---


## Estructura del proyecto

```text
ConsuGas/
│
├── data/
│   ├── modelo_cars.h5         #modelo MLP entrenado
│   ├── scaler_cars.pkl        #escalador de datos
│   └── features_cars.pkl      #columnas de entrenamiento
│
├── app.py                     #aplicación
├── requirements.txt           #dependencias
└── README.md

```
---
## Cómo correr

1. **Clonar repositorio y acceder**
```bash
git clone https://github.com/abbydoag/ProyectoFinalDL
cd ProyectoFinalDL
```
2. **Crear entorno virtual** (asegurarse de que sea en python 3.12 por Tensorflow)
```bash
python -m venv venv #para python 3.12: py -3.12 -m venv venv
venv\Scripts\activate

```
3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```
4. **Correr aplicación**
```bash
streamlit run app.py
```

---
## Cómo usar
1. **Ingresar perfil:** En la primera pestaña ingresar los datos que se solicitan del vehículo a evaluar (Cilindros, Tamaño del motor en litros, Caballos de fuerza y Peso en libras) y calcular predicción.
2. **Registrar cargas:** En cada carga de gasolina debemos guardar los siguientes datos e ingresarlos en la segunda pestaña:
    - Fecha
    - Trip (millas o kilómetros, no olvidar resetear medida antes de echar la gasolina)
    - Precio del galón
    - Monto total de carga

Se calculará cuantos galones esperaba el modelo y el rendimiento real en comparación con los galones realmente utilizados. Si se consumió más de lo esperado, la aplicación emitirá una alerta sugiriendo revisar el mantenimiento del vehículo o los hábitos de conducción.

---
## Detalles Técnicos del Modelo

- Dataset: Large Dataset of Cars (Kaggle). Se utilizó la fórmula oficial de la EPA (55% ciudad, 45% carretera) para calcular un rendimiento combinado.
- Arquitectura: Red Neuronal Densa (MLP) secuencial con capas de 64, 64 y 32 neuronas.
- Métricas de Evaluación:
    - RMSE (Root Mean Squared Error): 1.73 MPG.
    - MAE (Mean Absolute Error): 1.31 MPG.


---
# Laboratorio 10
El notebook del laboratorio está en la carpeta llamada 'Lab10', basicamente es el mismo notebook que el proyecto nada mas con lo necesario. En este ya se encuentran los resultados corridos. Tanto este como el notebook del proyecto fueron corridos en Google Colab por lo que recomiendo hacer lo mismo.

**Cómo ejecutar**
Simplemente colocar el notebook en google colab, preparar una carpeta para los datos csv del modelo y colocar el archivo, volver a ejecutar lo necesario.
