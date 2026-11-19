
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Detector de Conexiones",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #f8fafc;
}
.block-container {
    max-width: 1050px;
    padding-top: 2rem;
}
h1, h2, h3, p, label {
    color: #f8fafc !important;
}
[data-testid="stMetric"] {
    background-color: #17243a;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 14px;
}
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    border: none;
    padding: 12px 24px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    return joblib.load("mejor_modelo_svm_rbf.joblib")

st.title("🛡️ Detector Inteligente de Conexiones de Red")
st.markdown(
    "### Clasificación de conexiones mediante Machine Learning"
)
st.caption(
    "Proyecto académico | SVM con kernel RBF"
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Algoritmo", "SVM")
with col2:
    st.metric("Kernel", "RBF")
with col3:
    st.metric("Accuracy CV", "97.7 %")

st.divider()

st.subheader("📡 Análisis de una conexión")
st.write(
    "Introduce las características de la conexión "
    "para obtener una predicción."
)

col1, col2 = st.columns(2)

with col1:
    trafico = st.number_input(
        "Tráfico de red (Mbps)",
        min_value=0.0,
        max_value=500.0,
        value=50.0,
        step=1.0
    )

with col2:
    latencia = st.number_input(
        "Latencia (ms)",
        min_value=0.0,
        max_value=500.0,
        value=70.0,
        step=1.0
    )

if st.button("🔍 Analizar conexión", type="primary"):
    try:
        modelo = cargar_modelo()

        datos = pd.DataFrame({
            "trafico_mbps": [trafico],
            "latencia_ms": [latencia]
        })

        prediccion = modelo.predict(datos)[0]

        st.divider()
        st.subheader("📊 Resultado del análisis")

        if prediccion == 1:
            st.error("🚨 CONEXIÓN SOSPECHOSA")
            st.warning(
                "El modelo identifica un patrón sospechoso "
                "en los datos ingresados."
            )
        else:
            st.success("✅ CONEXIÓN NORMAL")
            st.info(
                "El modelo clasifica esta conexión como normal."
            )

    except Exception as error:
        st.error(f"Error al realizar la predicción: {error}")

st.divider()

st.subheader("🧠 Información del modelo")
st.write("""
**Algoritmo:** Máquina de Vectores de Soporte (SVM)

**Kernel:** Radial Basis Function (RBF)

**Parámetro C:** 10

**Gamma:** scale

**Variables utilizadas:**
- Tráfico de red (Mbps)
- Latencia de red (ms)

**Clasificación:**
- 0: Conexión normal
- 1: Conexión sospechosa
""")

st.caption(
    "Demostración académica con datos sintéticos. "
    "No sustituye un sistema real de ciberseguridad."
)


