import os
from dotenv import load_dotenv

load_dotenv() # Carga las variables de entorno del archivo .env

# --- Configuración de Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Configuración de Ollama (para modelos locales) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "qwen2.5-coder:1.5b") # Modelo por defecto

# --- Configuración de OpenRouter (para Claude y otros modelos) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")

# --- Configuración de Groq (Súper rápido) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# --- Modelos Recomendados (Gratis y Potentes) ---
MODELS_CONFIG = {
    "OPENROUTER": [
        {"name": "Gemini 2.0 Flash (Free)", "id": "google/gemini-2.0-flash-exp:free"},
        {"name": "Gemini 2.0 Pro (Free)", "id": "google/gemini-2.0-pro-exp-02-05:free"},
        {"name": "Mistral 7B (Free)", "id": "mistralai/mistral-7b-instruct:free"},
        {"name": "Meta Llama 3.1 405B (Free)", "id": "meta-llama/llama-3.1-405b-instruct:free"},
        {"name": "Claude 3.5 Sonnet (OpenRouter)", "id": "anthropic/claude-3.5-sonnet"},
        {"name": "Claude 3 Haiku (OpenRouter)", "id": "anthropic/claude-3-haiku"}
    ],
    "GROQ": [
        {"name": "Llama 3.3 70B (Versatile)", "id": "llama-3.3-70b-versatile"},
        {"name": "Llama 3.1 8B (Instant)", "id": "llama-3.1-8b-instant"},
        {"name": "Mixtral 8x7B (High Context)", "id": "mixtral-8x7b-32768"},
        {"name": "Gemma 2 9B", "id": "gemma2-9b-it"}
    ],
    "OLLAMA": [
        {"name": "Qwen 2.5 Coder 1.5B (Defecto)", "id": "qwen2.5-coder:1.5b"},
        {"name": "Llama 3.1 8B", "id": "llama3.1"},
        {"name": "Mistral", "id": "mistral"}
    ]
}

# --- Selector de Proveedor LLM ---
# Opciones: "GEMINI", "OLLAMA", "OPENROUTER", "GROQ"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "GEMINI")
