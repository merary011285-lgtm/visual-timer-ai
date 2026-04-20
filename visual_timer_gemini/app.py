import os
import sys
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import webbrowser
from threading import Timer
from main import generar_tabla_maestra, PERSONAJE_BLOQUE, ESTETICA_BLOQUE
from config import MODELS_CONFIG

def resource_path(relative_path):
    """ Obtiene la ruta absoluta para recursos, compatible con desarrollo y PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

app = Flask(__name__, 
            template_folder=resource_path('templates'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

# Asegurarse de que la carpeta de subidas exista
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', 
                         models_config=MODELS_CONFIG, 
                         personaje_bloque=PERSONAJE_BLOQUE, 
                         estetica_bloque=ESTETICA_BLOQUE)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return render_template('index.html', error="No se proporcionó ningún archivo de audio.")

    file = request.files['audio_file']

    if file.filename == '':
        return render_template('index.html', error="No se seleccionó ningún archivo.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Aquí llamamos a nuestra función principal
        try:
            # Capturamos la salida de la función en un string
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            redirected_output = sys.stdout = StringIO()

            llm_provider = request.form.get('llm_provider')
            model_name = request.form.get('model_name')
            p_bloque = request.form.get('personaje_bloque')
            e_bloque = request.form.get('estetica_bloque')

            generar_tabla_maestra(filepath, provider=llm_provider, model=model_name, personaje_bloque=p_bloque, estetica_bloque=e_bloque)
            result_table = redirected_output.getvalue()

            sys.stdout = old_stdout # Restaurar stdout
            # Intentar borrar el archivo temporal después de procesar; si no existe, registrar advertencia
            try:
                os.remove(filepath)
            except FileNotFoundError:
                app.logger.warning(f"Archivo no encontrado al intentar eliminar: {filepath}")
            except Exception as e:
                app.logger.error(f"Error al eliminar archivo {filepath}: {e}")
            # Fin de intento de borrado
            return render_template('index.html', 
                                 result=result_table, 
                                 provider=llm_provider, 
                                 model=model_name, 
                                 personaje_bloque=p_bloque, 
                                 estetica_bloque=e_bloque, 
                                 models_config=MODELS_CONFIG)
        except Exception as e:
            # Intentar borrar el archivo temporal incluso si ocurrió un error
            try:
                os.remove(filepath)
            except FileNotFoundError:
                app.logger.warning(f"Archivo no encontrado al intentar eliminar en el bloque except: {filepath}")
            except Exception as e:
                app.logger.error(f"Error al eliminar archivo en el bloque except: {filepath}: {e}")
            
            return render_template('index.html', 
                                 error=f"Error al procesar el audio: {e}", 
                                 models_config=MODELS_CONFIG,
                                 personaje_bloque=request.form.get('personaje_bloque'),
                                 estetica_bloque=request.form.get('estetica_bloque'))
    else:
        return render_template('index.html', error="Tipo de archivo no permitido. Solo MP3, WAV, FLAC.", models_config=MODELS_CONFIG)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['mp3', 'wav', 'flac']

@app.route('/status')
def check_status():
    provider = request.args.get('provider')
    from config import OLLAMA_BASE_URL, GEMINI_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY
    import requests

    try:
        if provider == "OLLAMA":
            # Verificar si el servidor local de Ollama responde
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            return {"status": "online" if r.status_code == 200 else "offline"}
        
        elif provider == "GEMINI":
            return {"status": "online" if GEMINI_API_KEY else "no_key"}
            
        elif provider == "OPENROUTER":
            return {"status": "online" if OPENROUTER_API_KEY else "no_key"}
            
        elif provider == "GROQ":
            return {"status": "online" if GROQ_API_KEY else "no_key"}
            
        return {"status": "unknown"}
    except:
        return {"status": "offline"}

if __name__ == '__main__':
    is_frozen = getattr(sys, 'frozen', False)
    if is_frozen:
        Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000/')).start()
    app.run(host='0.0.0.0', port=5000, debug=(not is_frozen))