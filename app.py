import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanSquaredError
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="ConsuGas", page_icon="⛽", layout="wide")

@st.cache_resource
def cargar_recursos():
    modelo = tf.keras.models.load_model('data/modelo_cars.h5', custom_objects={'mse': MeanSquaredError()})
    scaler = joblib.load('data/scaler_cars.pkl')
    features = joblib.load('data/features_cars.pkl')
    return modelo, scaler, features

modelo, scaler, features = cargar_recursos()

MILLA_A_KM = 1.60934 #conversion en caso use km

if 'historial_cargas' not in st.session_state:
    st.session_state.historial_cargas = []
if 'prediccion_mpg_actual' not in st.session_state:
    st.session_state.prediccion_mpg_actual = None

st.title("Sistema de Monitoreo de Combustible")
st.markdown("Predice el rendimiento de tu vehículo y registra cada carga para comparar con la realidad.")

sistema_unidades = st.radio("Selecciona la unidad:", ["Kilómetros", "Millas"], horizontal=True)
es_km = (sistema_unidades == "Kilómetros")
unidad_rendimiento = "km/galón" if es_km else "MPG"
unidad_consumo = "km" if es_km else "millas"

tab1, tab2 = st.tabs(["Perfil Vehículo", "Registro Cargas"])

with tab1:
    st.header("Perfil Vehículo")
    st.markdown("Poner los datos del vehículo que se piden.")
    
    col1, col2 = st.columns(2)
    with col1:
        cylinders = st.selectbox("Cilindros", [3, 4, 5, 6, 8], index=1)
        engine_size = st.number_input("Tamaño motor (Litros)", min_value=0.8, max_value=8.0, step=0.1)
        horsepower = st.number_input("Caballos de Fuerza (HP)", min_value=40, max_value=600, step=5)
        weight = st.number_input("Peso del vehículo (libras)", min_value=1500, max_value=6500, step=50)
        
    with col2:
        if st.button("Calcular Predicción", type="primary"):
            input_data = pd.DataFrame({
                'EngineSize': [engine_size], 'Cylinders': [cylinders],
                'HorsePower': [horsepower], 'Weight': [weight]
            })
            for col in features:
                if col not in input_data.columns: input_data[col] = 0
            input_data = input_data[features]

            input_scaled = scaler.transform(input_data)
            pred_mpg = modelo.predict(input_scaled)[0][0]
            
            st.session_state.prediccion_mpg_actual = pred_mpg * MILLA_A_KM if es_km else pred_mpg
            
            st.success(f"### Rendimiento Predicho por el Modelo: **{st.session_state.prediccion_mpg_actual:.2f} {unidad_rendimiento}**")
            st.markdown("Ahora registra tus cargas de gasolina.")

with tab2:
    st.header("Registro de Cargas")
    
    if st.session_state.prediccion_mpg_actual is None:
        st.warning("Calcula el rendimiento del auto")
    else:
        st.markdown(f"El modelo predice que harás **{st.session_state.prediccion_mpg_actual:.2f} {unidad_rendimiento}**.")
        with st.form("registro_carga"):
            st.subheader("Nueva Carga")
            st.markdown("Ingresa el Trip, el precio del galón y la cantidad de dinero que gastaste (sin necesidad de calcular los galones, el sistema lo hace).")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                fecha = st.date_input("Fecha de la carga", datetime.now())
                trip = st.number_input(f"Trip ({unidad_consumo})", min_value=1.0,)
            with col_r2:
                precio = st.number_input("Precio del galón (Q)", min_value=1.0, step=0.1)
            with col_r3:
                monto_total = st.number_input("Monto pagado (Q)", min_value=10.0, step=10.0)
            
            submit = st.form_submit_button("Registrar Carga")

        if submit:
            galones_reales = monto_total / precio
            rendimiento_real = trip / galones_reales
            galones_esperados = trip / st.session_state.prediccion_mpg_actual
            diferencia_galones = galones_reales - galones_esperados
            
            #hist
            st.session_state.historial_cargas.append({
                'Fecha': fecha.strftime("%Y-%m-%d"),
                'Trip': round(trip, 2),
                'Monto total(Q)': round(monto_total, 2),
                'Precio (Q)': round(precio, 2),
                'Galones Reales': round(galones_reales, 2),
                'Galones Esperados': round(galones_esperados, 2),
                'Rendimiento Real': round(rendimiento_real, 2)
            })
            st.success("Carga registrada correctamente.")

        
        if st.session_state.historial_cargas:
            st.markdown("---")
            st.subheader("Comparación de Datos Reales vs Predicción")
            
            df = pd.DataFrame(st.session_state.historial_cargas)
            ultima = df.iloc[-1]
            
            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.metric("Galones Reales Usados", f"{ultima['Galones Reales']} gal")
            with col_res2:
                st.metric("Galones Esperados", f"{ultima['Galones Esperados']:.2f} gal")
            with col_res3:
                st.metric("Rendimiento Real", f"{ultima['Rendimiento Real']} {unidad_rendimiento}")
                
            if ultima['Galones Reales'] > ultima['Galones Esperados']:
                st.error(f"En esta carga consumiste **{round(ultima['Galones Reales'] - ultima['Galones Esperados'], 2)} galones más** de lo que el modelo esperaba.")
            elif ultima['Galones Reales'] < ultima['Galones Esperados']:
                st.success(f"En esta carga consumiste **{round(ultima['Galones Esperados'] - ultima['Galones Reales'], 2)} galones menos** de lo que el modelo esperaba.")
            else:
                st.info("El consumo real fue exactamente igual al predicho por el modelo.")

            st.dataframe(df, use_container_width=True)

            #grafica
            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(len(df))
            width = 0.35
            ax.bar(x - width/2, df['Galones Esperados'], width, label='Galones Esperados', color='skyblue')
            ax.bar(x + width/2, df['Galones Reales'], width, label='Galones Reales', color='orange')
            
            ax.set_ylabel('Galones')
            ax.set_title('Consumo: Predicción vs Realidad por Carga')
            ax.set_xticks(x)
            ax.set_xticklabels(df['Fecha'])
            ax.legend()
            st.pyplot(fig)

st.markdown("---")
st.caption("Dataset: Large Dataset of Cars (Kaggle)")