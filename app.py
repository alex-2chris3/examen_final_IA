import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_page_config(
    page_title="Clasificador de Objetos",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Quitar margen superior */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.3rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 0.8rem;
    }

    .authors {
        text-align: center;
        font-size: 0.95rem;
        color: #e5e7eb;
        margin-bottom: 2rem;
    }

    .prediction-box {
        margin-top: 1.5rem;
        padding: 1.5rem 1.8rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #4b5563;
        box-shadow: 0 12px 30px rgba(0,0,0,0.45);
    }

    .pred-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #9ca3af;
        margin-bottom: 0.4rem;
    }

    .pred-class {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f9fafb;
        margin-bottom: 0.2rem;
    }

    .pred-confidence {
        font-size: 1rem;
        color: #d1d5db;
    }

    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.8rem;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cargamos el modelo
model = tf.keras.models.load_model("modelo_objetos_tf.h5", compile=False)
class_names = ["calculadora", "cartuchera", "engrapadora"]
img_size = (224, 224)

def preparar_imagen(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img = img.resize(img_size)

    arr = np.array(img).astype("float32") 
    arr = np.expand_dims(arr, 0)       
    return arr, img

# interfaz para el estilo 
st.markdown('<div class="main-title">Clasificador de Objetos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Proyecto de Inteligencia Artificial para reconocer una calculadora, cartuchera y engrapadora mediante imagenes</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="authors"><b>Autores:</b> Christian Hernández, Harold González, Pablo López</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Información")
    st.write(
        """
        Este prototipo usa el modelo MobileNet2
        entrenado con imágenes propias de:
        - Calculadoras  
        - Cartucheras  
        - Engrapadoras  
        """
    )
    st.write("Cargue una imagen para visualizar su clase y confianza de predicción.")
    st.caption("Desarrollado como proyecto de la materia Inteligencia Artificial")

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    x, img_display = preparar_imagen(uploaded_file)
    st.image(img_display, caption="Imagen subida", width=320)

    pred = model.predict(x)[0]
    idx = int(np.argmax(pred))
    confianza = float(pred[idx] * 100)

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="pred-label">Predicción</div>
            <div class="pred-class">{class_names[idx].upper()}</div>
            <div class="pred-confidence">Confianza: {confianza:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Probabilidades por clase")
    probs_dict = {class_names[i]: float(pred[i]) for i in range(len(class_names))}
    st.bar_chart(probs_dict)

st.markdown(
    '<div class="Gracias por su atencion.</div>',
    unsafe_allow_html=True,
)


#streamlit run app.py
