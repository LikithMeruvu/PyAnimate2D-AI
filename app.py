
import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame
import cv2
import numpy as np
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

# Initialize pygame
pygame.init()

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
    "You are a Pygame animation expert. Generate partial Python code for a 2D animation. "
    "The code must define two functions:\n"
    "  1) init_simulation(screen) -> None, for setting up any initial state.\n"
    "  2) update_and_draw(screen, dt) -> None, for updating the simulation & drawing one frame.\n\n"
    "Rules:\n"
    "• Do not create your own main loop. The host code calls your functions each frame.\n"
    "• You have access to these global variables: SCREEN_WIDTH, SCREEN_HEIGHT, FPS, DURATION, math, pygame.\n"
    "• All color args must be valid 3-tuples of integers (0-255), and you must handle window events properly if needed.\n"
    "• Do not prompt the user for any additional files, folders, or inputs.\n"
    "• Output only valid Python code with no markdown formatting or additional commentary.\n\n"
    "Important Duration Rule:\n"
    "• The user wants the simulation to run for exactly DURATION seconds. If your code logic ends early, "
    "  remain idle (e.g., static image) for the rest of the time. The final video must always match "
    "  fps * DURATION frames in length.\n"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, 'static', 'videos')
os.makedirs(VIDEO_DIR, exist_ok=True)
FRONTEND_VIDEO_DIR = os.path.join(BASE_DIR, 'static', 'frontend_videos')
os.makedirs(FRONTEND_VIDEO_DIR, exist_ok=True)

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
    code_block_match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', code, re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1)
    code = re.sub(r'```', '', code)
    lines = [line for line in code.splitlines() if line.strip() != ""]
    return "\n".join(lines)

def generate_animation_code(user_prompt: str, chat_session, duration) -> str:
    # Append duration info to the prompt
    user_prompt = user_prompt.strip() + f"\nDuration: {duration} seconds. The animation should run exactly for this time."
    response = chat_session.send_message(user_prompt)
    return response.text

def load_generated_simulation(code: str):
    header = (
        "import pygame\n"
        "import math\n"
        "import sys\n"
        "SCREEN_WIDTH=0\nSCREEN_HEIGHT=0\nFPS=30\nDURATION=30\n"
    )
    combined_code = header + "\n" + clean_generated_code(code)
    simulation_module = type(sys)('generated_simulation')
    sys.modules['generated_simulation'] = simulation_module
    exec(combined_code, simulation_module.__dict__)
    if 'init_simulation' not in simulation_module.__dict__:
        def init_simulation(screen):
            pass
        simulation_module.init_simulation = init_simulation
    if 'update_and_draw' not in simulation_module.__dict__:
        def update_and_draw(screen, dt):
            screen.fill((0, 0, 0))
            pygame.display.flip()
        simulation_module.update_and_draw = update_and_draw
    return simulation_module

def run_simulation(sim_module, video_filename, screen_size, fps, duration):
    sim_module.__dict__['SCREEN_WIDTH'] = screen_size[0]
    sim_module.__dict__['SCREEN_HEIGHT'] = screen_size[1]
    sim_module.__dict__['FPS'] = fps
    sim_module.__dict__['DURATION'] = duration

    pygame.display.set_mode(screen_size)
    screen = pygame.display.get_surface()
    sim_module.init_simulation(screen)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, screen_size)
    clock = pygame.time.Clock()
    total_frames = int(fps * duration)

    logger.debug(f"Starting simulation for {total_frames} frames.")
    try:
        for frame_idx in range(total_frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            dt = clock.get_time() / 1000.0
            sim_module.update_and_draw(screen, dt)
            arr = pygame.surfarray.array3d(screen)
            arr = arr.transpose((1, 0, 2))
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            video_writer.write(arr)
            clock.tick(fps)
            if frame_idx % 10 == 0:
                logger.debug(f"Frame {frame_idx} written.")
    finally:
        video_writer.release()
        cv2.destroyAllWindows()
        pygame.quit()
        del video_writer
        gc.collect()
        time.sleep(1)
        logger.debug("Simulation finished and file handles released.")

def run_generated_code(code: str, video_filename: str, duration: int, resolution=(1280, 720), fps=30):
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            sim_module = load_generated_simulation(code)
            run_simulation(sim_module, video_filename, resolution, fps, duration)
        return True, stdout_capture.getvalue()
    except Exception as e:
        msg = f"Error executing generated code:\n{str(e)}\n{traceback.format_exc()}"
        logger.error(msg)
        return False, msg

def copy_video_for_frontend(video_path: str) -> str:
    filename = os.path.basename(video_path)
    frontend_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
    
    attempts = 5
    for _ in range(attempts):
        try:
            shutil.copy2(video_path, frontend_path)
            time.sleep(1)
            if os.path.exists(frontend_path) and os.path.getsize(frontend_path) > 0:
                return filename
        except Exception as exc:
            logger.error(f"Error copying video: {exc}")
            time.sleep(1)
    raise RuntimeError("Failed to copy video to frontend directory.")

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

        quality = request.form.get("quality", "720p")
        resolution_map = {
            "144p": (256, 144),
            "240p": (426, 240),
            "360p": (640, 360),
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440)
        }
        resolution = resolution_map.get(quality, (1280, 720))

        try:
            fps = int(request.form.get("fps", 30))
        except ValueError:
            fps = 30

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
        if 'video_files' not in session:
            session['video_files'] = []

        max_attempts = 3
        final_success = False
        generated_code = ""
        error_output = ""
        output_path = ""
        for attempt in range(max_attempts):
            if attempt == 0:
                generated_code = generate_animation_code(prompt, chat_session, duration)
            else:
                fix_prompt = (
                    f"Previous code failed with error:\n{error_output}\n"
                    "Please fix it. Only output valid Python code with init_simulation & update_and_draw."
                )
                generated_code = generate_animation_code(fix_prompt, chat_session, duration)

            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            video_file_path = os.path.join(VIDEO_DIR, f"animation_{timestamp}.mp4")

            success, output = run_generated_code(
                generated_code, 
                video_file_path, 
                duration, 
                resolution, 
                fps
            )
            if success:
                final_success = True
                output_path = video_file_path
                break
            else:
                error_output = output

        if final_success:
            try:
                frontend_video_file = copy_video_for_frontend(output_path)
                session['chat_history'].insert(0, {
                    "role": "user",
                    "parts": [prompt]
                })
                session['chat_history'].insert(0, {
                    "role": "model",
                    "parts": [f"<pre><code>{generated_code}</code></pre>"]
                })
                session['video_files'].insert(0, frontend_video_file)
                session.modified = True

                return jsonify({
                    'success': True,
                    'video_file': frontend_video_file,
                    'code': generated_code,
                    'session_id': session_id
                })
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        else:
            return jsonify({'success': False, 'error': f"Failed to generate code after {max_attempts} attempts.\n{error_output}"})
    except Exception as ex:
        return jsonify({'success': False, 'error': str(ex)})

@app.route('/frontend_video/<path:filename>')
def frontend_video(filename):
    try:
        file_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
        if not os.path.exists(file_path):
            return '', 404

        file_size = os.path.getsize(file_path)
        range_header = request.headers.get('Range', None)
        if not range_header:
            with open(file_path, 'rb') as f:
                data = f.read()
            rv = Response(data, mimetype='video/mp4')
            rv.headers.add('Content-Length', str(file_size))
            return rv
        else:
            range_match = re.search(r'bytes=(\d+)-(\d*)', range_header)
            if not range_match:
                return '', 400
            start_str, end_str = range_match.groups()
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1
            length = end - start + 1
            with open(file_path, 'rb') as f:
                f.seek(start)
                data = f.read(length)
            rv = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
            rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
            rv.headers.add('Accept-Ranges', 'bytes')
            rv.headers.add('Content-Length', str(length))
            return rv
    except Exception as e:
        return '', 404

@app.route('/delete_video/<filename>', methods=['POST'])
def delete_video(filename):
    try:
        frontend_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
        original_path = os.path.join(VIDEO_DIR, filename)
        if os.path.exists(frontend_path):
            os.remove(frontend_path)
        if os.path.exists(original_path):
            os.remove(original_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

