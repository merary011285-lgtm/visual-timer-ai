import streamlit as st
import os
import sys
from io import StringIO
import time

# Importamos la lógica central de tu proyecto
from main import analizar_audio, generar_tabla_maestra, PERSONAJE_BLOQUE, ESTETICA_BLOQUE
from config import MODELS_CONFIG, OLLAMA_BASE_URL

# Configuración de la página
st.set_page_config(
    page_title="Visual Timer AI",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado (CSS simple de Streamlit)
st.markdown("""
    <style>
    .main {
        background-color: #0f0f10;
    }
    .stTextArea textarea {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("⏱️ Cronometrador Visual AI")
    st.subheader("Sincroniza rítmicamente tus prompts de video")

    # --- Barra Lateral ---
    if st.sidebar.button("🔄 Refrescar Pantalla"):
        st.rerun()

    st.sidebar.header("Configuración de IA")
    
    provider = st.sidebar.selectbox(
        "Proveedor de IA",
        ["GEMINI", "GROQ", "OPENROUTER", "OLLAMA"],
        index=0
    )

    # Filtrar modelos según proveedor
    available_models = MODELS_CONFIG.get(provider, [])
    model_names = [m["name"] for m in available_models]
    
    selected_model_name = st.sidebar.selectbox(
        "Modelos Sugeridos",
        ["-- Selección manual --"] + model_names
    )

    if selected_model_name == "-- Selección manual --":
        model_id = st.sidebar.text_input("ID del Modelo Manual", placeholder="ej: llama-3.1-70b")
    else:
        model_id = next(m["id"] for m in available_models if m["name"] == selected_model_name)
        st.sidebar.info(f"ID: {model_id}")

    st.sidebar.divider()

    # --- Bloques de Prompt ---
    st.sidebar.subheader("Estructura de Prompt")
    p_bloque = st.sidebar.text_area("Bloque Personaje", value=PERSONAJE_BLOQUE, height=150)
    e_bloque = st.sidebar.text_area("Bloque Estética", value=ESTETICA_BLOQUE, height=150)

    # --- Área Principal ---
    uploaded_file = st.file_uploader("Sube tu archivo de audio (MP3, WAV, FLAC)", type=["mp3", "wav", "flac"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.button("🚀 Generar Tabla Maestra", use_container_width=True):
            # Guardar archivo temporalmente
            with open("temp_audio.mp3", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.status("Analizando audio y generando prompts...", expanded=True) as status:
                st.write("🎵 Detectando beats y tempo...")
                
                # Redirigir stdout para capturar los prints de generar_tabla_maestra
                old_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                
                try:
                    # Llamada a tu función principal
                    generar_tabla_maestra(
                        "temp_audio.mp3", 
                        provider=provider, 
                        model=model_id, 
                        personaje_bloque=p_bloque, 
                        estetica_bloque=e_bloque
                    )
                    
                    resultado = mystdout.getvalue()
                    sys.stdout = old_stdout # Restaurar
                    
                    st.success("¡Sincronización completada!")
                    status.update(label="Proceso finalizado", state="complete", expanded=False)
                    
                    st.divider()
                    st.subheader("📋 Tabla de Sincronización")
                    st.code(resultado, language="markdown")
                    
                    # Botón de descarga
                    st.download_button(
                        label="💾 Descargar Tabla (.txt)",
                        data=resultado,
                        file_name="tabla_maestra_sincronizada.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"Error durante la generación: {e}")
                finally:
                    # Limpiar
                    if os.path.exists("temp_audio.mp3"):
                        os.remove("temp_audio.mp3")

    else:
        st.info("Esperando archivo de audio para comenzar...")

if __name__ == "__main__":
    main()
