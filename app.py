
import os
import google.generativeai as genai
from datetime import datetime
import sys
import io
import contextlib
import traceback
import re
import time
import math
import gc
import uuid
import shutil
from io import BytesIO
import mimetypes
import logging

from flask import Flask, render_template, request, session, jsonify, Response
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def get_vector_store(api_key):
    """
    Creates and returns a Chroma vector store for the p5.js documentation.
    If the vector store already exists, it will be loaded from disk.
    """
    persist_directory = "./chroma_db"

    # Configure the generative AI client
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(persist_directory):
        # Load the existing vector store
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        # Create the vector store from scratch
        loader = WebBaseLoader("https://p5js.org/reference/")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)

        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )

    return vector_store

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# Constants & Available Models
# ----------------------------
DEFAULT_MODEL_NAME = "gemini-2.0-flash-thinking-exp-01-21"
DEFAULT_TEMPERATURE = 0.9
MAX_DURATION = 60  # seconds

# List of available Gemini models
AVAILABLE_MODELS = [
    {"name": "Gemini 2.0 Flash", "id": "gemini-2.0-flash-001"},
    {"name": "Gemini 2.0 Pro", "id": "gemini-2.0-pro-exp-02-05"},
    {"name": "Gemini 2.0 Flash-Lite", "id": "gemini-2.0-flash-lite-preview-02-05"},
    {"name": "Gemini 2.0 Flash Thinking", "id": "gemini-2.0-flash-thinking-exp-01-21"},
    {"name": "Gemini 1.5 Flash", "id": "gemini-1.5-flash"},
    {"name": "Gemini 1.5 Pro", "id": "gemini-1.5-pro"},
    {"name": "Gemini 1.0 Pro", "id": "gemini-1.0-pro"},
    {"name": "Gemini 1.0 Pro Vision", "id": "gemini-1.0-pro-vision"}
]

SYSTEM_PROMPT = (
    "You are a p5.js animation expert. Generate JavaScript code for a 2D animation. "
    "The code must define two functions:\n"
    "  1) setup() -> Must call createCanvas(width, height) and set up the initial state.\n"
    "  2) draw() -> Called each frame to update and draw the animation.\n\n"
    "Rules:\n"
    "• Do not create your own p5.js instance (e.g., `new p5(...)`). The host environment will do this.\n"
    "• You have access to these global variables: `width`, `height` (the canvas dimensions).\n"
    "• You have access to all standard p5.js functions (e.g., `rect`, `ellipse`, `background`, `fill`, etc.).\n"
    "• All color arguments must be valid p5.js color values.\n"
    "• Output only valid JavaScript code with no markdown formatting or additional commentary.\n"
)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.static_folder = 'static'
app.static_url_path = '/static'

# In-memory dictionary to hold chat sessions per user (keyed by session id)
chat_sessions = {}

# ----------------------------
# Simulation Generation
# ----------------------------
def clean_generated_code(code: str) -> str:
    """Cleans up the generated code from the AI model."""
    code_block_match = re.search(r'```(?:javascript)?\s*([\s\S]*?)\s*```', code, re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1)
    code = re.sub(r'```', '', code)
    lines = [line for line in code.splitlines() if line.strip() != ""]
    return "\n".join(lines)

def generate_animation_code(user_prompt: str, chat_session, duration) -> str:
    """Generates animation code using the generative AI model."""
    user_prompt = user_prompt.strip() + f"\nDuration: {duration} seconds. The animation should run exactly for this time."
    response = chat_session.send_message(user_prompt)
    return response.text

# ----------------------------
# Flask Routes
# ----------------------------
@app.route("/")
def index():
    config = {
        "api_key": session.get("api_key", ""),
        "model_name": session.get("model_name", DEFAULT_MODEL_NAME),
        "temperature": session.get("temperature", DEFAULT_TEMPERATURE)
    }
    session_id = session.get('session_id', '')
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'video_files' not in session:
        session['video_files'] = []
    return render_template(
        "index.html", 
        chat_history=session['chat_history'], 
        video_files=session['video_files'],
        session_id=session_id,
        config=config,
        available_models=AVAILABLE_MODELS
    )

@app.route("/set_config", methods=["POST"])
def set_config():
    api_key = request.form.get("api_key", "").strip()
    model_name = request.form.get("model_name", DEFAULT_MODEL_NAME)
    try:
        temp = float(request.form.get("temperature", DEFAULT_TEMPERATURE))
        if temp < 0.0: temp = 0.0
        if temp > 1.0: temp = 1.0
    except ValueError:
        temp = DEFAULT_TEMPERATURE

    session["api_key"] = api_key
    session["model_name"] = model_name
    session["temperature"] = temp

    return jsonify({"success": True})

@app.route("/new_session", methods=["POST"])
def new_session():
    # Clear the chat session and chat history
    session.pop("session_id", None)
    session['chat_history'] = []
    return jsonify({"success": True})

@app.route("/generate", methods=["POST"])
def generate():
    try:
        prompt = request.form.get("prompt", "")
        user_api_key = request.form.get("api_key", "").strip()
        model_name = request.form.get("model_name", DEFAULT_MODEL_NAME)
        try:
            temperature = float(request.form.get("temperature", DEFAULT_TEMPERATURE))
        except ValueError:
            temperature = DEFAULT_TEMPERATURE

        # Enforce 0.0 <= temperature <= 1.0
        if temperature < 0.0: temperature = 0.0
        if temperature > 1.0: temperature = 1.0

        session["api_key"] = user_api_key
        session["model_name"] = model_name
        session["temperature"] = temperature

        if not user_api_key:
            return jsonify({
                'success': False,
                'error': 'Please enter your Gemini API key (0.0 to 1.0 temp).'
            })

        try:
            genai.configure(api_key=user_api_key)
        except Exception:
            return jsonify({
                'success': False,
                'error': 'Invalid API key. Please provide a valid Gemini API key.'
            })

        try:
            duration = int(request.form.get("duration", 30))
        except ValueError:
            duration = 30
        duration = min(duration, MAX_DURATION)

        # Get the vector store
        vector_store = get_vector_store(user_api_key)

        # Retrieve relevant documents
        retriever = vector_store.as_retriever()
        docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in docs])

        # Create a new prompt with the context
        enhanced_prompt = (
            "Based on the following p5.js documentation, create an animation:\n\n"
            f"Documentation:\n{context}\n\n"
            f"User Prompt: {prompt}"
        )

        session_id = request.form.get("session_id", "")
        if session_id and session_id in chat_sessions:
            chat_session = chat_sessions[session_id]
        else:
            session_id = str(uuid.uuid4())
            model_instance = genai.GenerativeModel(
                model_name,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(temperature=temperature)
            )
            chat_session = model_instance.start_chat(history=[])
            chat_sessions[session_id] = chat_session
            session['session_id'] = session_id

        if 'chat_history' not in session:
            session['chat_history'] = []

        generated_code = generate_animation_code(enhanced_prompt, chat_session, duration)
        cleaned_code = clean_generated_code(generated_code)

        session['chat_history'].insert(0, {"role": "user", "parts": [prompt]})
        session['chat_history'].insert(0, {"role": "model", "parts": [cleaned_code]})
        session.modified = True

        return jsonify({
            'success': True,
            'code': cleaned_code,
            'session_id': session_id
        })

    except Exception as ex:
        return jsonify({'success': False, 'error': str(ex)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
