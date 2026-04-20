import google.generativeai as genai
from pydub import AudioSegment
from pydub.utils import mediainfo
import math
import librosa
import soundfile as sf
import json
import requests # <--- Nuevo para Ollama
from config import GEMINI_API_KEY, OLLAMA_BASE_URL, LOCAL_MODEL, LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_MODEL, GROQ_API_KEY, GROQ_MODEL

# --- Prompt General ---
PROMPT_TEMPLATE = """
Eres un Director de Video AI especializado en Edición Rítmica Estricta.
Tu misión es generar la descripción de un plano de video continuo y coherente, así como datos de estado, para un segmento de audio.

Debes respetar estrictamente los siguientes bloques inmutables para garantizar la identidad visual:

[BLOQUE PERSONAJE]:
{personaje_bloque}

[BLOQUE ESTÉTICA]:
{estetica_bloque}

Información del Segmento de Audio:
- Tiempo: {tiempo_inicio} - {tiempo_fin}
- Tempo General (BPM): {tempo_audio} (Usa esto para inferir la energía y el "mood" de la canción para la Intensidad).

Contexto del Plano Anterior (para asegurar la continuidad narrativa):
{contexto_anterior}

Instrucciones para la Generación del Plano:
1.  **Acción Narrativa (El Prompt Único):** Describe la acción principal o el "relleno" visual. Puede ser un momento clave, una reacción emocional, o un detalle atmosférico. Asegúrate de que [BLOQUE PERSONAJE] y [BLOQUE ESTÉTICA] se incluyan al inicio y al final de esta descripción.
2.  **Movimiento de Cámara:** Usa un léxico cinematográfico detallado (ej., Dolly In, Crane Shot Ascending, Handheld Jittery Follow, Steadicam Smooth Push In, Zoom Dolly, Pan Right, Tilt Up, Tracking Shot).
3.  **Intensidad:** Refleja la dinámica del segmento de audio (Baja, Media, Alta, Máxima).

Formato de Salida (JSON, asegúrate de que sea un JSON válido y escapado si es necesario):
```json
{{
    "accion_narrativa": "[Tu descripción del plano, incluyendo bloques inmutables]",
    "movimiento_camara": "[Tu descripción del movimiento de cámara]",
    "intensidad": "[Tu descripción de intensidad]",
    "estado_siguiente": {{
        "posicion_personaje": "[ej. primer plano, plano medio, fondo]",
        "accion_principal": "[ej. camina, detiene, mira, corre, salta]",
        "angulo_camara_final": "[ej. picado, contrapicado, nivel de los ojos]",
        "movimiento_camara_final": "[ej. static, zoom, pan, tracking]",
        "emocion_dominante": "[ej. misterio, determinación, furia, calma]",
        "objetos_foco": "[ej. un medallón, un arma, un horizonte]"
    }}
}}
```
"""

def parse_json_response(raw_text: str, default_personaje: str, default_estetica: str) -> dict:
    """Extrae y parsea el bloque JSON de una respuesta de texto."""
    try:
        json_start = raw_text.find('```json')
        json_end = raw_text.rfind('```')

        if json_start != -1 and json_end != -1 and json_start < json_end:
            json_str = raw_text[json_start + 7 : json_end].strip()
        else:
            json_str = raw_text.strip()
            # Intentar limpiar si hay texto antes/después
            if '{' in json_str:
                json_str = json_str[json_str.find('{'):json_str.rfind('}')+1]

        response_json = json.loads(json_str)
        return {
            "accion_narrativa": response_json.get("accion_narrativa", "Acción narrativa no generada."),
            "movimiento_camara": response_json.get("movimiento_camara", "Movimiento de cámara no generado."),
            "intensidad": response_json.get("intensidad", "Intensidad no generada."),
            "estado_siguiente": response_json.get("estado_siguiente", {})
        }
    except Exception as e:
        print(f"ERROR: No se pudo parsear JSON: {e}\nTexto: {raw_text}")
        return {
            "accion_narrativa": f"{default_personaje} + Error de parseo JSON + {default_estetica}",
            "movimiento_camara": "Error",
            "intensidad": "Error",
            "estado_siguiente": {}
        }

# --- Configuración de LLM (Gemini o Ollama) ---
MODELO_GEMINI = None

if LLM_PROVIDER == "GEMINI":
    genai.configure(api_key=GEMINI_API_KEY)
    MODELO_GEMINI = genai.GenerativeModel('models/gemini-pro-latest')
    print("DEBUG: Usando el proveedor LLM: GEMINI")
elif LLM_PROVIDER == "OLLAMA":
    print(f"DEBUG: Usando el proveedor LLM: OLLAMA con URL: {OLLAMA_BASE_URL} y modelo: {LOCAL_MODEL}")
elif LLM_PROVIDER == "OPENROUTER":
    print(f"DEBUG: Usando el proveedor LLM: OPENROUTER con modelo: {OPENROUTER_MODEL}")
elif LLM_PROVIDER == "GROQ":
    print(f"DEBUG: Usando el proveedor LLM: GROQ con modelo: {GROQ_MODEL}")
else:
    raise ValueError("Proveedor LLM no válido en config.py")

# --- Bloques Inmutables (Ejemplo) ---
PERSONAJE_BLOQUE = """
[BLOQUE PERSONAJE]: Descripción física exacta (Raza, edad, ropa, peinado, accesorios).
"""

ESTETICA_BLOQUE = """
[BLOQUE ESTÉTICA]: Descripción técnica exacta (Cámara, Lente, LUT de color, Estilo de render).
"""

def analizar_audio(audio_path: str):
    """
    Analiza un archivo de audio para obtener su duración, tempo (BPM) y los tiempos de los beats.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None) # Cargar audio, mantener la tasa de muestreo original
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        
        # Detección de tempo y beats
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        return {
            "duration_ms": duration_seconds * 1000,
            "duration_seconds": duration_seconds,
            "tempo": tempo,
            "beat_times": beat_times.tolist() # Convertir a lista para fácil manejo
        }
    except Exception as e:
        print(f"Error al analizar el audio {audio_path}: {e}")
        return None

def _generar_contenido_gemini(tiempo_inicio: str, tiempo_fin: str, personaje_bloque: str, estetica_bloque: str, tempo_audio: float, contexto_anterior: str = "") -> dict:
    prompt_texto = PROMPT_TEMPLATE.format(
        tiempo_inicio=tiempo_inicio,
        tiempo_fin=tiempo_fin,
        personaje_bloque=personaje_bloque,
        estetica_bloque=estetica_bloque,
        tempo_audio=tempo_audio,
        contexto_anterior=contexto_anterior
    )
    
    response = MODELO_GEMINI.generate_content(prompt_texto)
    return parse_json_response(response.text, personaje_bloque, estetica_bloque)

import time

def generar_contenido_llm(tiempo_inicio: str, tiempo_fin: str, personaje_bloque: str, estetica_bloque: str, tempo_audio: float, contexto_anterior: str = "", provider_override=None, model_override=None) -> dict:
    """
    Función envoltorio que decide qué proveedor LLM usar para generar contenido.
    Añade lógica de reintento para errores 429 (Rate Limit).
    """
    provider = provider_override or LLM_PROVIDER
    max_retries = 3
    retry_delay = 5 # segundos iniciales

    for attempt in range(max_retries):
        try:
            if provider == "GEMINI":
                return _generar_contenido_gemini(tiempo_inicio, tiempo_fin, personaje_bloque, estetica_bloque, tempo_audio, contexto_anterior)
            elif provider == "OLLAMA":
                current_model = model_override or LOCAL_MODEL
                return _generar_contenido_ollama(tiempo_inicio, tiempo_fin, personaje_bloque, estetica_bloque, tempo_audio, contexto_anterior, model=current_model)
            elif provider == "OPENROUTER":
                current_model = model_override or OPENROUTER_MODEL
                return _generar_contenido_openrouter(tiempo_inicio, tiempo_fin, personaje_bloque, estetica_bloque, tempo_audio, contexto_anterior, model=current_model)
            elif provider == "GROQ":
                current_model = model_override or GROQ_MODEL
                # Pequeño sleep preventivo para Groq que es muy estricto
                time.sleep(1.5)
                return _generar_contenido_groq(tiempo_inicio, tiempo_fin, personaje_bloque, estetica_bloque, tempo_audio, contexto_anterior, model=current_model)
            else:
                raise ValueError(f"Proveedor LLM '{provider}' no válido configurado.")
        
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                print(f"DEBUG: Rate limit detectado (429). Reintentando en {retry_delay}s... (Intento {attempt + 1})")
                time.sleep(retry_delay)
                retry_delay *= 2 # Backoff exponencial
                continue
            else:
                # Si no es un 429 o ya agotamos intentos, lanzamos el error original
                print(f"ERROR en intento {attempt + 1}: {e}")
                # Devolver un objeto de error para que la tabla no se rompa
                return {
                    "accion_narrativa": f"{personaje_bloque} + Error tras reintentos + {estetica_bloque}",
                    "movimiento_camara": "Error",
                    "intensidad": "Error",
                    "estado_siguiente": {}
                }

def generar_prompt_video(tiempo_inicio: str, tiempo_fin: str, accion_narrativa: str, movimiento_camara: str, intensidad: str) -> str:
    """
    Genera el prompt completo para un bloque de video.
    """
    return f"{tiempo_inicio} - {tiempo_fin}|{accion_narrativa}|{movimiento_camara}|{intensidad}"

def _generar_contenido_groq(tiempo_inicio: str, tiempo_fin: str, personaje_bloque: str, estetica_bloque: str, tempo_audio: float, contexto_anterior: str = "", model=None) -> dict:
    prompt_texto = PROMPT_TEMPLATE.format(
        tiempo_inicio=tiempo_inicio,
        tiempo_fin=tiempo_fin,
        personaje_bloque=personaje_bloque,
        estetica_bloque=estetica_bloque,
        tempo_audio=tempo_audio,
        contexto_anterior=contexto_anterior
    )
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model or GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt_texto}]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data, timeout=120)
    response.raise_for_status()
    raw_response = response.json()["choices"][0]["message"]["content"]
    return parse_json_response(raw_response, personaje_bloque, estetica_bloque)

def _generar_contenido_ollama(tiempo_inicio: str, tiempo_fin: str, personaje_bloque: str, estetica_bloque: str, tempo_audio: float, contexto_anterior: str = "", model=None) -> dict:
    prompt_texto = PROMPT_TEMPLATE.format(
        tiempo_inicio=tiempo_inicio,
        tiempo_fin=tiempo_fin,
        personaje_bloque=personaje_bloque,
        estetica_bloque=estetica_bloque,
        tempo_audio=tempo_audio,
        contexto_anterior=contexto_anterior
    )
    
    headers = {'Content-Type': 'application/json'}
    data = {
        "model": model or LOCAL_MODEL,
        "prompt": prompt_texto,
        "stream": False
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", headers=headers, json=data, timeout=120)
    response.raise_for_status()
    raw_response = response.json().get("response", "")
    return parse_json_response(raw_response, personaje_bloque, estetica_bloque)

def _generar_contenido_openrouter(tiempo_inicio: str, tiempo_fin: str, personaje_bloque: str, estetica_bloque: str, tempo_audio: float, contexto_anterior: str = "", model=None) -> dict:
    prompt_texto = PROMPT_TEMPLATE.format(
        tiempo_inicio=tiempo_inicio,
        tiempo_fin=tiempo_fin,
        personaje_bloque=personaje_bloque,
        estetica_bloque=estetica_bloque,
        tempo_audio=tempo_audio,
        contexto_anterior=contexto_anterior
    )
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model or OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt_texto}]
    }
    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=120)
    response.raise_for_status()
    raw_response = response.json()["choices"][0]["message"]["content"]
    return parse_json_response(raw_response, personaje_bloque, estetica_bloque)

def generar_tabla_maestra(audio_path: str, provider: str = None, model: str = None, personaje_bloque: str = None, estetica_bloque: str = None):
    """
    Genera la tabla maestra de prompts de video basada en el análisis del audio.
    """
    # Usar bloques por defecto si no se proporcionan
    p_bloque = personaje_bloque or PERSONAJE_BLOQUE
    e_bloque = estetica_bloque or ESTETICA_BLOQUE

    print(f"DEBUG: Iniciando generar_tabla_maestra con provider={provider}...")
    audio_info = analizar_audio(audio_path)
    if audio_info is None:
        print("No se pudo analizar el archivo de audio.")
        return

    duracion_ms = audio_info["duration_ms"]
    duracion_segundos = audio_info["duration_seconds"]
    tempo = audio_info["tempo"]
    beat_times = audio_info["beat_times"]
    
    print("I. DATOS GLOBALES (Copiar y Pegar)") # DEBUG
    print(f"Prompt Base (Estilo): {e_bloque}")
    print(f"Personaje (Anchor): {p_bloque}")
    print(f"Tempo (BPM): {tempo.item():.2f}")
    print("\nII. SECUENCIA DE TIEMPO (SHOT LIST 5s)\n")
    print("Tiempo|Acción Narrativa (El Prompt Único)|Movimiento de Cámara|Intensidad")
    print("---|---|---|---")

    intervalo_base = 5 # segundos
    
    # --- Modelo de Estado para Continuidad Narrativa ---
    estado_escena = {
        "posicion_personaje": "iniciando en plano general", 
        "accion_principal": "neutral",
        "angulo_camara_final": "general",
        "movimiento_camara_final": "static",
        "emocion_dominante": "expectante",
        "objetos_foco": "ninguno"
    }
    
    segment_end_times = [0.0] # Iniciar con el tiempo 0
    
    current_time = 0.0
    while current_time < duracion_segundos:
        target_time = current_time + intervalo_base
        
        # Encontrar el beat más cercano al target_time
        closest_beat = None
        min_diff = float('inf')
        for beat in beat_times:
            if beat > current_time: # Solo beats después del tiempo actual
                diff = abs(beat - target_time)
                if diff < min_diff:
                    min_diff = diff
                    closest_beat = beat
        
        # Encontrar el beat más cercano al target_time
        # Si encontramos un beat cercano (ej. dentro de 1.5 segundo del target_time), lo usamos
        # De lo contrario, usamos el intervalo_base fijo
        if closest_beat is not None and min_diff <= 1.5: # Umbral de 1.5 segundos para enganchar al beat
            end_time_for_segment = closest_beat
        else:
            end_time_for_segment = target_time

        # Asegurarse de no exceder la duración total del audio
        end_time_for_segment = min(end_time_for_segment, duracion_segundos)
        
        # Asegurarse de que el tiempo final sea mayor que el tiempo actual para evitar segmentos de 0 duración
        if end_time_for_segment > current_time:
            segment_end_times.append(end_time_for_segment)
        
        current_time = end_time_for_segment
    
    # Eliminar beats duplicados o muy cercanos si los hubiera (para evitar segmentos muy cortos)
    final_segment_times = [segment_end_times[0]]
    for i in range(1, len(segment_end_times)):
        if segment_end_times[i] - final_segment_times[-1] > 1.0: # Asegurar que el segmento sea de al menos 1 segundo
            final_segment_times.append(segment_end_times[i])

    # Asegurarse de que el último segmento llegue hasta el final del audio
    if final_segment_times[-1] < duracion_segundos:
        final_segment_times.append(duracion_segundos)

    # Ahora iteramos sobre los tiempos de los segmentos calculados
    for i in range(len(final_segment_times) - 1):
        inicio_seg = final_segment_times[i]
        fin_seg = final_segment_times[i+1]

        # Formatear tiempos para los prompts
        tiempo_inicio_str = f"{int(inicio_seg // 60):02d}:{int(inicio_seg % 60):02d}"
        tiempo_fin_str = f"{int(fin_seg // 60):02d}:{int(fin_seg % 60):02d}"

        # Convertir estado_escena a una cadena para pasarla a Gemini
        contexto_anterior_str = (
            f"El plano anterior terminó con:\n"
            f"- Posición del personaje: {estado_escena['posicion_personaje']}\n"
            f"- Acción principal: {estado_escena['accion_principal']}\n"
            f"- Ángulo de cámara: {estado_escena['angulo_camara_final']}\n"
            f"- Movimiento de cámara: {estado_escena['movimiento_camara_final']}\n"
            f"- Emoción dominante: {estado_escena['emocion_dominante']}\n"
            f"- Objetos en foco: {estado_escena['objetos_foco']}."
            f"Continúa la narrativa y la acción desde este punto, generando prompts de relleno si es apropiado."
        )

        # Asegurarse de que el tempo sea un float plano
        tempo_float = float(tempo) if not hasattr(tempo, 'item') else tempo.item()

        gemini_output = generar_contenido_llm(
            tiempo_inicio_str, 
            tiempo_fin_str, 
            p_bloque, 
            e_bloque,
            tempo_float, # Pasar el tempo como un float
            contexto_anterior=contexto_anterior_str, # Pasando el estado como contexto anterior
            provider_override=provider,
            model_override=model
        )
        # Debug robusto para evitar errores de slice
        acc_narrativa = str(gemini_output.get('accion_narrativa', ''))
        print(f"DEBUG: Contenido generado para {tiempo_inicio_str} - {tiempo_fin_str}: {{'accion_narrativa': '{acc_narrativa[:50]}...'}}")

        accion_narrativa = gemini_output["accion_narrativa"]
        movimiento_camara = gemini_output["movimiento_camara"]
        intensidad = gemini_output["intensidad"]
        
        # --- Actualizar el estado de la escena para el siguiente segmento usando la respuesta JSON de Gemini ---
        estado_escena.update(gemini_output["estado_siguiente"])
        print(f"DEBUG: Estado de escena actualizado: {estado_escena}") # DEBUG
        
        print(generar_prompt_video(tiempo_inicio_str, tiempo_fin_str, accion_narrativa, movimiento_camara, intensidad))

# --- Ejecución ---
# if __name__ == "__main__":

#     audio_de_ejemplo = r"Luz en la Oscuridad.mp3" 
    
#     # 2. Descomentamos la llamada a la función:
#     generar_tabla_maestra(audio_de_ejemplo)

#     # 3. (Opcional) Puedes comentar los prints informativos anteriores para que no ensucien la salida
#     # print("Para ejecutar, crea un archivo de audio...")

#     print("DEBUG: Script main.py iniciado.") # DEBUG inicial
