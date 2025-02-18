# import os
# os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

# import pygame
# import cv2
# import numpy as np
# import google.generativeai as genai
# from datetime import datetime
# import sys
# import io
# import contextlib
# import traceback
# import re
# import time
# import math
# import gc
# import uuid
# import shutil
# from io import BytesIO
# import mimetypes
# import logging

# from flask import Flask, render_template, request, session, jsonify, Response
# from flask import send_from_directory
# from flask import request  # needed for range header

# # Initialize pygame
# pygame.init()

# # Set up logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# # ----------------------------
# # Configuration & Gemini Setup
# # ----------------------------
# GEMINI_API_KEY = "AIzaSyDB5RaCIs-rr5XMHfwQ-rjXczYJY9Wxfdw"  # Replace with your actual API key
# genai.configure(api_key=GEMINI_API_KEY)

# DEFAULT_MODEL_NAME = "gemini-2.0-pro-exp-02-05"
# DEFAULT_TEMPERATURE = 0.9
# MAX_DURATION = 60  # seconds

# SYSTEM_PROMPT = (
#     "You are a Pygame animation expert. Generate partial Python code for a 2D animation. "
#     "The code must define two functions:\n"
#     "  1) init_simulation(screen) -> None, for setting up any initial state.\n"
#     "  2) update_and_draw(screen, dt) -> None, for updating the simulation & drawing one frame.\n\n"
#     "Rules:\n"
#     "• Do not create your own main loop. The host code calls your functions each frame.\n"
#     "• You have access to these global variables: SCREEN_WIDTH, SCREEN_HEIGHT, FPS, DURATION, math, pygame.\n"
#     "• All color args must be valid 3-tuples of integers (0-255), and you must handle window events properly if needed.\n"
#     "• Do not prompt the user for any additional files, folders, or inputs.\n"
#     "• Output only valid Python code with no markdown formatting or additional commentary.\n\n"
#     "Important Duration Rule:\n"
#     "• The user wants the simulation to run for exactly DURATION seconds. If your code logic ends early, "
#     "  remain idle (e.g., static image) for the rest of the time. The final video must always match "
#     "  fps * DURATION frames in length.\n"
# )

# VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/videos')
# os.makedirs(VIDEO_DIR, exist_ok=True)
# FRONTEND_VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/frontend_videos')
# os.makedirs(FRONTEND_VIDEO_DIR, exist_ok=True)
# CHUNK_SIZE = 8192

# app = Flask(__name__)
# app.secret_key = os.urandom(24)
# app.static_folder = 'static'
# app.static_url_path = '/static'

# chat_sessions = {}

# # ----------------------------
# # Simulation Generation Functions
# # ----------------------------
# def clean_generated_code(code: str) -> str:
#     code_block_match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', code, re.IGNORECASE)
#     if code_block_match:
#         code = code_block_match.group(1)
#     code = re.sub(r'```', '', code)
#     lines = [line for line in code.splitlines() if line.strip() != ""]
#     return "\n".join(lines)

# def generate_animation_code(user_prompt: str, chat_session) -> str:
#     response = chat_session.send_message(user_prompt)
#     return response.text

# def load_generated_simulation(code: str):
#     header = (
#         "import pygame\n"
#         "import math\n"
#         "import sys\n"
#         "SCREEN_WIDTH=0\nSCREEN_HEIGHT=0\nFPS=30\nDURATION=30\n"
#     )
#     combined_code = header + "\n" + clean_generated_code(code)
#     simulation_module = type(sys)('generated_simulation')
#     sys.modules['generated_simulation'] = simulation_module
#     exec(combined_code, simulation_module.__dict__)
#     if 'init_simulation' not in simulation_module.__dict__:
#         def init_simulation(screen):
#             pass
#         simulation_module.init_simulation = init_simulation
#     if 'update_and_draw' not in simulation_module.__dict__:
#         def update_and_draw(screen, dt):
#             screen.fill((0, 0, 0))
#             pygame.display.flip()
#         simulation_module.update_and_draw = update_and_draw
#     return simulation_module

# def run_simulation(sim_module, video_filename, screen_size, fps, duration):
#     sim_module.__dict__['SCREEN_WIDTH'] = screen_size[0]
#     sim_module.__dict__['SCREEN_HEIGHT'] = screen_size[1]
#     sim_module.__dict__['FPS'] = fps
#     sim_module.__dict__['DURATION'] = duration

#     pygame.display.set_mode(screen_size)
#     screen = pygame.display.get_surface()
#     sim_module.init_simulation(screen)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video_writer = cv2.VideoWriter(video_filename, fourcc, fps, screen_size)
#     clock = pygame.time.Clock()
#     total_frames = int(fps * duration)

#     logger.debug(f"Starting simulation for {total_frames} frames.")
#     try:
#         for frame_idx in range(total_frames):
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     break
#             dt = clock.get_time() / 1000.0
#             sim_module.update_and_draw(screen, dt)
#             arr = pygame.surfarray.array3d(screen)
#             arr = arr.transpose((1, 0, 2))
#             arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
#             video_writer.write(arr)
#             clock.tick(fps)
#             if frame_idx % 10 == 0:
#                 logger.debug(f"Frame {frame_idx} written.")
#     finally:
#         video_writer.release()
#         cv2.destroyAllWindows()
#         pygame.quit()
#         del video_writer
#         gc.collect()
#         # Give extra time for Windows to release file handles
#         time.sleep(1)
#         logger.debug("Simulation finished and file handles released.")

# def run_generated_code(code: str, video_filename: str, duration: int, resolution=(1280,720), fps=30):
#     stdout_capture = io.StringIO()
#     try:
#         with contextlib.redirect_stdout(stdout_capture):
#             sim_module = load_generated_simulation(code)
#             run_simulation(sim_module, video_filename, resolution, fps, duration)
#         return True, stdout_capture.getvalue()
#     except Exception as e:
#         msg = f"Error executing generated code:\n{str(e)}\n{traceback.format_exc()}"
#         logger.error(msg)
#         return False, msg

# def verify_video_exists(filepath: str, timeout: int = 10) -> bool:
#     start_time = time.time()
#     while time.time() - start_time < timeout:
#         if os.path.exists(filepath):
#             try:
#                 with open(filepath, 'rb') as f:
#                     first_bytes = f.read(8192)
#                     if first_bytes:
#                         return True
#             except (IOError, OSError):
#                 pass
#         time.sleep(0.5)
#     return False

# def copy_video_for_frontend(video_path: str) -> str:
#     filename = os.path.basename(video_path)
#     frontend_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
    
#     logger.debug(f"Attempting to copy video from {video_path} to {frontend_path}")
#     if not verify_video_exists(video_path):
#         raise RuntimeError(f"Source video not ready: {video_path}")
    
#     os.makedirs(FRONTEND_VIDEO_DIR, exist_ok=True)
    
#     max_attempts = 5
#     for attempt in range(max_attempts):
#         try:
#             shutil.copy2(video_path, frontend_path)
#             time.sleep(1)
#             if verify_video_exists(frontend_path):
#                 logger.debug("Video copy verified.")
#                 return filename
#         except (IOError, OSError) as e:
#             logger.error(f"Copy attempt {attempt + 1} failed: {e}")
#             time.sleep(1)
    
#     raise RuntimeError("Failed to copy and verify video")

# # ----------------------------
# # Flask Routes (UI & API)
# # ----------------------------
# @app.route("/", methods=["GET"])
# def index():
#     if 'chat_history' not in session:
#         session['chat_history'] = []
#     if 'video_files' not in session:
#         session['video_files'] = []
#     return render_template("index.html", 
#                            chat_history=session['chat_history'], 
#                            video_files=session['video_files'])

# @app.route("/generate", methods=["POST"])
# def generate():
#     try:
#         logger.info("Starting video generation")
#         prompt = request.form.get("prompt")
#         model_name = request.form.get("model_name") or DEFAULT_MODEL_NAME
#         try:
#             temperature = float(request.form.get("temperature") or DEFAULT_TEMPERATURE)
#         except ValueError:
#             temperature = DEFAULT_TEMPERATURE

#         try:
#             duration = int(request.form.get("duration") or 30)
#         except ValueError:
#             duration = 30
#         if duration > MAX_DURATION:
#             duration = MAX_DURATION

#         quality = request.form.get("quality")
#         resolution_map = {
#             "144p": (256, 144),
#             "240p": (426, 240),
#             "360p": (640, 360),
#             "480p": (854, 480),
#             "720p": (1280, 720),
#             "1080p": (1920, 1080),
#             "1440p": (2560, 1440)
#         }
#         resolution = resolution_map.get(quality, (1280, 720))

#         try:
#             fps = int(request.form.get("fps") or 30)
#         except ValueError:
#             fps = 30

#         session_id = request.form.get("session_id")
#         if session_id and session_id in chat_sessions:
#             chat_session = chat_sessions[session_id]
#         else:
#             session_id = str(uuid.uuid4())
#             model_instance = genai.GenerativeModel(
#                 model_name,
#                 system_instruction=SYSTEM_PROMPT,
#                 generation_config=genai.GenerationConfig(temperature=temperature)
#             )
#             chat_session = model_instance.start_chat(history=[])
#             chat_sessions[session_id] = chat_session

#         if 'video_files' not in session:
#             session['video_files'] = []
#         if 'chat_history' not in session:
#             session['chat_history'] = []

#         max_attempts = 3
#         final_success = False
#         generated_code = ""
#         error_output = ""
#         output_log = ""
#         output_path = ""
#         for attempt in range(max_attempts):
#             if attempt == 0:
#                 generated_code = generate_animation_code(prompt, chat_session)
#             else:
#                 fix_prompt = (
#                     f"Previous code failed with error:\n{error_output}\n"
#                     "Please fix it. Only output valid Python code with init_simulation & update_and_draw. "
#                     "No extra inputs or files."
#                 )
#                 generated_code = generate_animation_code(fix_prompt, chat_session)
#             timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#             video_file_path = os.path.join(VIDEO_DIR, f"animation_{timestamp}.mp4")
#             logger.info(f"Attempting generation with file: {video_file_path}")
#             success, output = run_generated_code(generated_code, video_file_path, duration, resolution, fps)
#             if success:
#                 final_success = True
#                 output_log = output
#                 output_path = video_file_path
#                 break
#             else:
#                 error_output = output

#         if final_success:
#             try:
#                 frontend_video_file = copy_video_for_frontend(output_path)
#                 logger.info(f"Video saved to: {output_path}")
#                 logger.info(f"Frontend video file: {frontend_video_file}")
                
#                 session['chat_history'].insert(0, {
#                     "role": "user", 
#                     "parts": [prompt]
#                 })
#                 session['chat_history'].insert(0, {
#                     "role": "model", 
#                     "parts": [f"<pre><code>{generated_code}</code></pre>"]
#                 })
#                 session['video_files'].insert(0, frontend_video_file)
#                 session.modified = True
                
#                 return jsonify({
#                     'success': True,
#                     'video_file': frontend_video_file,
#                     'code': generated_code,
#                 })

#             except Exception as e:
#                 logger.error(f"Error in final stage: {str(e)}", exc_info=True)
#                 return jsonify({'success': False, 'error': str(e)})
#     except Exception as e:
#         logger.error(f"Unexpected error in generate route: {str(e)}", exc_info=True)
#         return jsonify({'success': False, 'error': str(e)})

# # New video route that supports Range requests and then closes file handles immediately
# @app.route('/frontend_video/<path:filename>')
# def frontend_video(filename):
#     try:
#         file_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
#         logger.info(f"Serving frontend video from: {file_path}")
#         if not os.path.exists(file_path):
#             logger.error(f"File not found: {file_path}")
#             return '', 404

#         file_size = os.path.getsize(file_path)
#         logger.info(f"File size: {file_size} bytes")
#         range_header = request.headers.get('Range', None)
#         if not range_header:
#             logger.info("No Range header; serving full file")
#             with open(file_path, 'rb') as f:
#                 data = f.read()
#             rv = Response(data, mimetype='video/mp4')
#             rv.headers.add('Content-Length', str(file_size))
#             return rv
#         else:
#             logger.info(f"Range header: {range_header}")
#             range_match = re.search(r'bytes=(\d+)-(\d*)', range_header)
#             if range_match:
#                 start_str, end_str = range_match.groups()
#                 start = int(start_str)
#                 end = int(end_str) if end_str and end_str.strip() != "" else file_size - 1
#             else:
#                 logger.error("Invalid Range header format")
#                 return '', 400
#             length = end - start + 1
#             logger.info(f"Serving bytes: {start}-{end} (length {length})")
#             with open(file_path, 'rb') as f:
#                 f.seek(start)
#                 data = f.read(length)
#             rv = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
#             rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
#             rv.headers.add('Accept-Ranges', 'bytes')
#             rv.headers.add('Content-Length', str(length))
#             return rv
#     except Exception as e:
#         logger.error(f"Error serving frontend video: {e}", exc_info=True)
#         return '', 404

# @app.route('/delete_video/<filename>', methods=['POST'])
# def delete_video(filename):
#     try:
#         frontend_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
#         original_path = os.path.join(VIDEO_DIR, filename)
#         logger.info(f"Attempting to delete video: {filename}")
#         if os.path.exists(frontend_path):
#             try:
#                 os.remove(frontend_path)
#                 logger.info(f"Deleted frontend copy: {frontend_path}")
#             except Exception as ex:
#                 logger.error(f"Failed to delete frontend copy: {ex}", exc_info=True)
#                 return jsonify({'success': False, 'error': str(ex)})
#         if os.path.exists(original_path):
#             try:
#                 os.remove(original_path)
#                 logger.info(f"Deleted original video: {original_path}")
#             except Exception as ex:
#                 logger.error(f"Failed to delete original video: {ex}", exc_info=True)
#                 return jsonify({'success': False, 'error': str(ex)})
#         return jsonify({'success': True})
#     except Exception as e:
#         logger.error(f"Error deleting video: {e}", exc_info=True)
#         return jsonify({'success': False, 'error': str(e)})

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)


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
from flask import send_from_directory

# Initialize pygame
pygame.init()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration & Gemini Setup
# ----------------------------
GEMINI_API_KEY = "AIzaSyDB5RaCIs-rr5XMHfwQ-rjXczYJY9Wxfdw"  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_MODEL_NAME = "gemini-2.0-pro-exp-02-05"
DEFAULT_TEMPERATURE = 0.9
MAX_DURATION = 60  # seconds

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

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/videos')
os.makedirs(VIDEO_DIR, exist_ok=True)
FRONTEND_VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/frontend_videos')
os.makedirs(FRONTEND_VIDEO_DIR, exist_ok=True)
CHUNK_SIZE = 8192

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.static_folder = 'static'
app.static_url_path = '/static'

chat_sessions = {}

# ----------------------------
# Simulation Generation Functions
# ----------------------------
def clean_generated_code(code: str) -> str:
    code_block_match = re.search(r'```(?:python)?\s*([\s\S]*?)\s*```', code, re.IGNORECASE)
    if code_block_match:
        code = code_block_match.group(1)
    code = re.sub(r'```', '', code)
    lines = [line for line in code.splitlines() if line.strip() != ""]
    return "\n".join(lines)

def generate_animation_code(user_prompt: str, chat_session) -> str:
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

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # or *"H264"
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
        time.sleep(1)  # extra delay for Windows to release file handles
        logger.debug("Simulation finished and file handles released.")

def run_generated_code(code: str, video_filename: str, duration: int, resolution=(1280,720), fps=30):
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

def verify_video_exists(filepath: str, timeout: int = 10) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    first_bytes = f.read(8192)
                    if first_bytes:
                        return True
            except (IOError, OSError):
                pass
        time.sleep(0.5)
    return False

def copy_video_for_frontend(video_path: str) -> str:
    filename = os.path.basename(video_path)
    frontend_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
    
    logger.debug(f"Attempting to copy video from {video_path} to {frontend_path}")
    if not verify_video_exists(video_path):
        raise RuntimeError(f"Source video not ready: {video_path}")
    
    os.makedirs(FRONTEND_VIDEO_DIR, exist_ok=True)
    
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            shutil.copy2(video_path, frontend_path)
            time.sleep(1)
            if verify_video_exists(frontend_path):
                logger.debug("Video copy verified.")
                return filename
        except (IOError, OSError) as e:
            logger.error(f"Copy attempt {attempt + 1} failed: {e}")
            time.sleep(1)
    
    raise RuntimeError("Failed to copy and verify video")

# ----------------------------
# Flask Routes (UI & API)
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'video_files' not in session:
        session['video_files'] = []
    return render_template("index.html", 
                           chat_history=session['chat_history'], 
                           video_files=session['video_files'])

@app.route("/generate", methods=["POST"])
def generate():
    try:
        logger.info("Starting video generation")
        prompt = request.form.get("prompt")
        model_name = request.form.get("model_name") or DEFAULT_MODEL_NAME
        try:
            temperature = float(request.form.get("temperature") or DEFAULT_TEMPERATURE)
        except ValueError:
            temperature = DEFAULT_TEMPERATURE

        try:
            duration = int(request.form.get("duration") or 30)
        except ValueError:
            duration = 30
        if duration > MAX_DURATION:
            duration = MAX_DURATION

        quality = request.form.get("quality")
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
            fps = int(request.form.get("fps") or 30)
        except ValueError:
            fps = 30

        session_id = request.form.get("session_id")
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

        if 'video_files' not in session:
            session['video_files'] = []
        if 'chat_history' not in session:
            session['chat_history'] = []

        max_attempts = 3
        final_success = False
        generated_code = ""
        error_output = ""
        output_log = ""
        output_path = ""
        for attempt in range(max_attempts):
            if attempt == 0:
                generated_code = generate_animation_code(prompt, chat_session)
            else:
                fix_prompt = (
                    f"Previous code failed with error:\n{error_output}\n"
                    "Please fix it. Only output valid Python code with init_simulation & update_and_draw. "
                    "No extra inputs or files."
                )
                generated_code = generate_animation_code(fix_prompt, chat_session)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            video_file_path = os.path.join(VIDEO_DIR, f"animation_{timestamp}.mp4")
            logger.info(f"Attempting generation with file: {video_file_path}")
            success, output = run_generated_code(generated_code, video_file_path, duration, resolution, fps)
            if success:
                final_success = True
                output_log = output
                output_path = video_file_path
                break
            else:
                error_output = output

        if final_success:
            try:
                frontend_video_file = copy_video_for_frontend(output_path)
                logger.info(f"Video saved to: {output_path}")
                logger.info(f"Frontend video file: {frontend_video_file}")
                
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
                })

            except Exception as e:
                logger.error(f"Error in final stage: {str(e)}", exc_info=True)
                return jsonify({'success': False, 'error': str(e)})
    except Exception as e:
        logger.error(f"Unexpected error in generate route: {str(e)}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

# Video route that supports Range requests
@app.route('/frontend_video/<path:filename>')
def frontend_video(filename):
    try:
        file_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
        logger.info(f"Serving frontend video from: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return '', 404

        file_size = os.path.getsize(file_path)
        logger.info(f"File size: {file_size} bytes")
        range_header = request.headers.get('Range', None)
        if not range_header:
            logger.info("No Range header; serving full file")
            with open(file_path, 'rb') as f:
                data = f.read()
            rv = Response(data, mimetype='video/mp4')
            rv.headers.add('Content-Length', str(file_size))
            return rv
        else:
            logger.info(f"Range header: {range_header}")
            range_match = re.search(r'bytes=(\d+)-(\d*)', range_header)
            if range_match:
                start_str, end_str = range_match.groups()
                start = int(start_str)
                end = int(end_str) if end_str and end_str.strip() != "" else file_size - 1
            else:
                logger.error("Invalid Range header format")
                return '', 400
            length = end - start + 1
            logger.info(f"Serving bytes: {start}-{end} (length {length})")
            with open(file_path, 'rb') as f:
                f.seek(start)
                data = f.read(length)
            rv = Response(data, 206, mimetype='video/mp4', direct_passthrough=True)
            rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
            rv.headers.add('Accept-Ranges', 'bytes')
            rv.headers.add('Content-Length', str(length))
            return rv
    except Exception as e:
        logger.error(f"Error serving frontend video: {e}", exc_info=True)
        return '', 404

@app.route('/delete_video/<filename>', methods=['POST'])
def delete_video(filename):
    try:
        frontend_path = os.path.join(FRONTEND_VIDEO_DIR, filename)
        original_path = os.path.join(VIDEO_DIR, filename)
        logger.info(f"Attempting to delete video: {filename}")
        if os.path.exists(frontend_path):
            try:
                os.remove(frontend_path)
                logger.info(f"Deleted frontend copy: {frontend_path}")
            except Exception as ex:
                logger.error(f"Failed to delete frontend copy: {ex}", exc_info=True)
                return jsonify({'success': False, 'error': str(ex)})
        if os.path.exists(original_path):
            try:
                os.remove(original_path)
                logger.info(f"Deleted original video: {original_path}")
            except Exception as ex:
                logger.error(f"Failed to delete original video: {ex}", exc_info=True)
                return jsonify({'success': False, 'error': str(ex)})
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error deleting video: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
