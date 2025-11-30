import streamlit as st
import tempfile
import os
import cv2
import numpy as np
import time
import mediapipe as mp
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from collections import Counter
import re
import threading
import queue
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import wave
from concurrent.futures import ThreadPoolExecutor
import logging

# Set page config
st.set_page_config(
    page_title="Educational Assistant for Students with Learning Disabilities",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize speech recognition globally
recognizer = sr.Recognizer()
audio_queue = None

# Initialize session state variables if they don't exist
if 'distraction_count' not in st.session_state:
    st.session_state.distraction_count = 0
if 'distraction_events' not in st.session_state:
    st.session_state.distraction_events = []
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'live_transcript' not in st.session_state:
    st.session_state.live_transcript = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'important_points' not in st.session_state:
    st.session_state.important_points = []
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'stop_recording' not in st.session_state:
    st.session_state.stop_recording = False
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'audio_chunks' not in st.session_state:
    st.session_state.audio_chunks = []
if 'recording_complete' not in st.session_state:
    st.session_state.recording_complete = False
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()
if 'mic_initialized' not in st.session_state:
    st.session_state.mic_initialized = False
if 'level_percentage' not in st.session_state:
    st.session_state.level_percentage = 0
if 'recording_duration' not in st.session_state:
    st.session_state.recording_duration = 0
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None

# New parallel mode session state variables
if 'parallel_active' not in st.session_state:
    st.session_state.parallel_active = False
if 'distraction_thread' not in st.session_state:
    st.session_state.distraction_thread = None
if 'speech_thread' not in st.session_state:
    st.session_state.speech_thread = None
if 'parallel_stop_event' not in st.session_state:
    st.session_state.parallel_stop_event = threading.Event()
if 'video_frame_queue' not in st.session_state:
    st.session_state.video_frame_queue = queue.Queue(maxsize=5)
if 'transcription_queue' not in st.session_state:
    st.session_state.transcription_queue = queue.Queue()
if 'parallel_results' not in st.session_state:
    st.session_state.parallel_results = {
        'current_frame': None,
        'current_transcript': "",
        'distraction_status': "Monitoring...",
        'recording_status': "Listening...",
        'distraction_count': 0,
        'distraction_events': []
    }

# Title and description
st.title("Educational Assistant for Students with Learning Disabilities")
st.markdown("""
This application helps students stay focused and understand content through three main features:
1. **Distraction Detection**: Monitors for lack of attention through webcam
2. **Speech Processing**: Transcribes and summarizes speech or audio files
3. **Parallel Mode**: Runs both features simultaneously for comprehensive monitoring
""")

# Create tabs for the three main modules
tab1, tab2, tab3 = st.tabs(["Distraction Detection", "Speech Processing", "Parallel Mode"])

# === Face Detection Helper Functions ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=3)
mp_drawing = mp.solutions.drawing_utils

# Constants for distraction detection
FACE_TURN_THRESHOLD = 2
EYE_CLOSED_THRESHOLD = 4
EAR_THRESHOLD = 0.25
FRAME_SMOOTHING_WINDOW = 5
MIN_FACE_SIZE = 10000

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP = 1

def calculate_ear(eye):
    """Calculate eye aspect ratio"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def get_largest_face(faces, w, h):
    """Find the largest face in the frame"""
    max_area = 0
    best = None
    for face in faces:
        x = [lm.x * w for lm in face.landmark]
        y = [lm.y * h for lm in face.landmark]
        area = (max(x) - min(x)) * (max(y) - min(y))
        if area > max_area and area > MIN_FACE_SIZE:
            max_area = area
            best = face
    return best

def process_frame_thread_safe(frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, distraction_count, distraction_events):
    """Process a single frame for distraction detection - thread safe version"""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    distraction_detected = False
    distraction_type = None
    now = time.time()
    output_frame = frame.copy()

    if results.multi_face_landmarks:
        face = get_largest_face(results.multi_face_landmarks, w, h)
        if face:
            landmarks = face.landmark
            nose_x = landmarks[NOSE_TIP].x * w
            center_x = w / 2
            deviation = abs(nose_x - center_x)
            face_dir_buffer.append(deviation)
            if len(face_dir_buffer) > FRAME_SMOOTHING_WINDOW:
                face_dir_buffer.pop(0)
            avg_dev = sum(face_dir_buffer) / len(face_dir_buffer)

            if avg_dev > 80:
                if face_turn_start is None:
                    face_turn_start = now
                elif now - face_turn_start > FACE_TURN_THRESHOLD:
                    distraction_detected = True
                    distraction_type = "Face turned"
            else:
                face_turn_start = None

            left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE])
            right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE])
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            ear_buffer.append(ear)
            if len(ear_buffer) > FRAME_SMOOTHING_WINDOW:
                ear_buffer.pop(0)
            avg_ear = sum(ear_buffer) / len(ear_buffer)

            if avg_ear < EAR_THRESHOLD:
                if eye_close_start is None:
                    eye_close_start = now
                elif now - eye_close_start > EYE_CLOSED_THRESHOLD:
                    distraction_detected = True
                    distraction_type = "Eyes closed"
            else:
                eye_close_start = None

            mp_drawing.draw_landmarks(
                output_frame, 
                face, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    if distraction_detected and not in_distraction:
        distraction_count += 1
        in_distraction = True
        last_distraction_time = now
        distraction_events.append(f"{distraction_type} at {time.strftime('%H:%M:%S')}")

    if not distraction_detected and in_distraction:
        if (now - last_distraction_time) > 2:
            in_distraction = False

    # Add text overlay
    cv2.putText(output_frame, f'Distractions: {distraction_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add status indicator for attention
    status = "Focused" if not distraction_detected else f"Distracted: {distraction_type}"
    color = (0, 255, 0) if not distraction_detected else (0, 0, 255)
    cv2.putText(output_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return output_frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, distraction_detected, distraction_type, distraction_count, distraction_events

def process_frame(frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time):
    """Process a single frame for distraction detection - original version for single tab use"""
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    distraction_detected = False
    distraction_type = None
    now = time.time()
    output_frame = frame.copy()

    if results.multi_face_landmarks:
        face = get_largest_face(results.multi_face_landmarks, w, h)
        if face:
            landmarks = face.landmark
            nose_x = landmarks[NOSE_TIP].x * w
            center_x = w / 2
            deviation = abs(nose_x - center_x)
            face_dir_buffer.append(deviation)
            if len(face_dir_buffer) > FRAME_SMOOTHING_WINDOW:
                face_dir_buffer.pop(0)
            avg_dev = sum(face_dir_buffer) / len(face_dir_buffer)

            if avg_dev > 80:
                if face_turn_start is None:
                    face_turn_start = now
                elif now - face_turn_start > FACE_TURN_THRESHOLD:
                    distraction_detected = True
                    distraction_type = "Face turned"
            else:
                face_turn_start = None

            left_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE])
            right_eye = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE])
            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            ear_buffer.append(ear)
            if len(ear_buffer) > FRAME_SMOOTHING_WINDOW:
                ear_buffer.pop(0)
            avg_ear = sum(ear_buffer) / len(ear_buffer)

            if avg_ear < EAR_THRESHOLD:
                if eye_close_start is None:
                    eye_close_start = now
                elif now - eye_close_start > EYE_CLOSED_THRESHOLD:
                    distraction_detected = True
                    distraction_type = "Eyes closed"
            else:
                eye_close_start = None

            mp_drawing.draw_landmarks(
                output_frame, 
                face, 
                mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
            )

    if distraction_detected and not in_distraction:
        st.session_state.distraction_count += 1
        in_distraction = True
        last_distraction_time = now
        st.session_state.distraction_events.append(f"{distraction_type} at {time.strftime('%H:%M:%S')}")

    if not distraction_detected and in_distraction:
        if (now - last_distraction_time) > 2:
            in_distraction = False

    # Add text overlay
    cv2.putText(output_frame, f'Distractions: {st.session_state.distraction_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Add status indicator for attention
    status = "Focused" if not distraction_detected else f"Distracted: {distraction_type}"
    color = (0, 255, 0) if not distraction_detected else (0, 0, 255)
    cv2.putText(output_frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return output_frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, distraction_detected, distraction_type

# === Parallel Processing Functions ===
def distraction_detection_thread(stop_event, frame_queue, results_dict):
    """Thread function for distraction detection"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            results_dict['distraction_status'] = "Error: Cannot access webcam"
            return
        
        ear_buffer, face_dir_buffer = [], []
        face_turn_start = eye_close_start = None
        in_distraction = False
        last_distraction_time = 0
        
        # Thread-local distraction tracking
        thread_distraction_count = 0
        thread_distraction_events = []
        
        results_dict['distraction_status'] = "Webcam initialized - Monitoring..."
        
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                results_dict['distraction_status'] = "Error: Failed to capture frame"
                break
            
            # Process frame for distraction detection using thread-safe version
            processed_frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, distraction_detected, distraction_type, thread_distraction_count, thread_distraction_events = process_frame_thread_safe(
                frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, thread_distraction_count, thread_distraction_events
            )
            
            # Update results dictionary with thread-safe data
            results_dict['distraction_count'] = thread_distraction_count
            results_dict['distraction_events'] = thread_distraction_events.copy()
            
            # Update status
            if distraction_detected:
                results_dict['distraction_status'] = f"🔴 DISTRACTED: {distraction_type}"
            else:
                results_dict['distraction_status'] = "✅ FOCUSED"
            
            # Put frame in queue for display (non-blocking)
            try:
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                if not frame_queue.full():
                    frame_queue.put(rgb_frame, block=False)
            except queue.Full:
                pass  # Skip frame if queue is full
            
            # Small delay to control frame rate
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        results_dict['distraction_status'] = "Webcam monitoring stopped"
        
    except Exception as e:
        results_dict['distraction_status'] = f"Error in distraction detection: {str(e)}"

def speech_processing_thread(stop_event, transcription_queue, results_dict):
    """Thread function for speech processing"""
    try:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        # Adjust for ambient noise
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        results_dict['recording_status'] = "🎤 Microphone ready - Listening..."
        accumulated_transcript = ""
        
        while not stop_event.is_set():
            try:
                with mic as source:
                    # Listen for audio with shorter timeout for responsiveness
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                    # Check if should stop before processing
                    if stop_event.is_set():
                        break
                    
                    # Try to transcribe
                    try:
                        text = recognizer.recognize_google(audio)
                        if text:
                            accumulated_transcript += " " + text
                            # Put transcription in queue
                            try:
                                transcription_queue.put(text, block=False)
                            except queue.Full:
                                pass
                            
                            results_dict['recording_status'] = f"🎙️ Transcribing: '{text[:30]}...'"
                            results_dict['current_transcript'] = accumulated_transcript.strip()
                    except sr.UnknownValueError:
                        results_dict['recording_status'] = "🔍 Waiting for clear speech..."
                    except sr.RequestError as e:
                        results_dict['recording_status'] = f"❌ Recognition error: {str(e)}"
                        break
                        
            except sr.WaitTimeoutError:
                if not stop_event.is_set():
                    results_dict['recording_status'] = "⏳ Waiting for speech..."
                continue
                
    except Exception as e:
        results_dict['recording_status'] = f"❌ Speech processing error: {str(e)}"

def start_parallel_monitoring():
    """Start parallel monitoring with both threads"""
    if st.session_state.parallel_active:
        return
    
    # Reset stop event and results
    st.session_state.parallel_stop_event.clear()
    st.session_state.parallel_results = {
        'current_frame': None,
        'current_transcript': "",
        'distraction_status': "Initializing...",
        'recording_status': "Initializing...",
        'distraction_count': 0,
        'distraction_events': []
    }
    
    # Clear queues
    while not st.session_state.video_frame_queue.empty():
        try:
            st.session_state.video_frame_queue.get_nowait()
        except queue.Empty:
            break
    
    while not st.session_state.transcription_queue.empty():
        try:
            st.session_state.transcription_queue.get_nowait()
        except queue.Empty:
            break
    
    # Start threads
    st.session_state.distraction_thread = threading.Thread(
        target=distraction_detection_thread,
        args=(st.session_state.parallel_stop_event, st.session_state.video_frame_queue, st.session_state.parallel_results),
        daemon=True
    )
    
    st.session_state.speech_thread = threading.Thread(
        target=speech_processing_thread,
        args=(st.session_state.parallel_stop_event, st.session_state.transcription_queue, st.session_state.parallel_results),
        daemon=True
    )
    
    # Start both threads
    st.session_state.distraction_thread.start()
    st.session_state.speech_thread.start()
    
    st.session_state.parallel_active = True

def stop_parallel_monitoring():
    """Stop parallel monitoring"""
    if not st.session_state.parallel_active:
        return
    
    # Signal threads to stop
    st.session_state.parallel_stop_event.set()
    
    # Wait for threads to finish (with timeout)
    if st.session_state.distraction_thread and st.session_state.distraction_thread.is_alive():
        st.session_state.distraction_thread.join(timeout=2.0)
    
    if st.session_state.speech_thread and st.session_state.speech_thread.is_alive():
        st.session_state.speech_thread.join(timeout=2.0)
    
    # Transfer thread results to session state for display
    if 'distraction_count' in st.session_state.parallel_results:
        st.session_state.distraction_count = st.session_state.parallel_results['distraction_count']
    if 'distraction_events' in st.session_state.parallel_results:
        st.session_state.distraction_events = st.session_state.parallel_results['distraction_events']
    
    st.session_state.parallel_active = False
    st.session_state.distraction_thread = None
    st.session_state.speech_thread = None

# === Text Processing Helper Functions ===
def summarize_text(text):
    """Summarize the provided text"""
    if not text or len(text) < 50:  # Skip if text is too short
        return "Text too short for summarization", []
    
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
        summaries = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk) < 50:  # Skip chunks that are too short
                continue
            with st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}..."):
                summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                summaries.append(summary)
        
        final_summary = " ".join(summaries)
        
        # Extract important points
        words = re.findall(r'\w+', final_summary.lower())
        word_counts = Counter(words)
        # Filter out common words
        common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'that', 'this', 'it', 'its'])
        filtered_words = {word: count for word, count in word_counts.items() if word not in common_words and len(word) > 3}
        top_words = [word for word, _ in sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        # Get sentences containing important words
        sentences = [s.strip() for s in re.split(r'[.!?]', final_summary) if s.strip()]
        important_points = []
        
        for sentence in sentences:
            if any(word in sentence.lower() for word in top_words):
                if sentence not in important_points:
                    important_points.append(sentence)
            if len(important_points) >= 5:  # Limit to 5 key points
                break
        
        return final_summary, important_points
    except Exception as e:
        st.error(f"Error in summarization: {e}")
        return str(e), []

def record_and_transcribe_live():
    """Record audio and transcribe in real-time using speech_recognition with proper stop functionality"""
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    
    try:
        # Initialize microphone
        mic = sr.Microphone()
        
        # Adjust for ambient noise
        with mic as source:
            st.write("🎤 **Adjusting for background noise... Please wait.**")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.write("🎤 **Ready! Start speaking...**")
    except Exception as e:
        st.error(f"Microphone setup error: {e}")
        return
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    transcription_placeholder = st.empty()
    duration_placeholder = st.empty()
    
    # Recording loop with proper stop checking
    try:
        while st.session_state.is_recording:
            # Update recording duration
            if st.session_state.recording_start_time:
                duration = time.time() - st.session_state.recording_start_time
                duration_placeholder.markdown(f"⏱️ **Recording Duration: {int(duration//60):02d}:{int(duration%60):02d}**")
            
            status_placeholder.markdown("🎙️ **Listening...**")
            
            with mic as source:
                try:
                    # Listen for audio with shorter timeout for responsiveness
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                    # Check if recording should stop before processing
                    if not st.session_state.is_recording:
                        break
                    
                    # Try to transcribe the audio
                    try:
                        text = recognizer.recognize_google(audio)
                        if text:
                            st.session_state.live_transcript += " " + text
                            # Update the transcription display
                            transcription_placeholder.text_area(
                                "Live Transcription:", 
                                st.session_state.live_transcript.strip(),
                                height=200,
                                key=f"live_transcript_{len(st.session_state.live_transcript)}"
                            )
                    except sr.UnknownValueError:
                        # No speech detected, continue listening
                        status_placeholder.markdown("🔍 **Waiting for clear speech...**")
                    except sr.RequestError as e:
                        st.error(f"Google Speech Recognition error: {e}")
                        break
                        
                except sr.WaitTimeoutError:
                    # Timeout occurred, continue the loop if still recording
                    if st.session_state.is_recording:
                        status_placeholder.markdown("⏳ **Waiting for speech...**")
                        continue
                    else:
                        break
                        
    except Exception as e:
        st.error(f"Recording error: {e}")
    finally:
        # Clean up when recording stops
        st.session_state.is_recording = False
        status_placeholder.markdown("✅ **Recording stopped**")
        
        # Update final duration
        if st.session_state.recording_start_time:
            final_duration = time.time() - st.session_state.recording_start_time
            duration_placeholder.markdown(f"⏱️ **Total Recording Duration: {int(final_duration//60):02d}:{int(final_duration%60):02d}**")
        
        # Show final transcription
        if st.session_state.live_transcript:
            transcription_placeholder.text_area(
                "Final Transcription:", 
                st.session_state.live_transcript.strip(),
                height=200,
                disabled=True,
                key="final_transcript_display"
            )

def process_uploaded_audio(uploaded_file):
    """Process uploaded audio file and return transcription"""
    import speech_recognition as sr
    import tempfile
    import os
    
    recognizer = sr.Recognizer()
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name
        
        # Transcribe the audio file
        with sr.AudioFile(temp_path) as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.record(source)
            
        # Get transcription
        text = recognizer.recognize_google(audio)
        
        # Clean up temporary file
        os.unlink(temp_path)
        
        return text
        
    except sr.UnknownValueError:
        return "Could not understand the audio. Please try again with clearer speech."
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error processing audio file: {e}"
    finally:
        # Ensure temporary file is cleaned up
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

# === Distraction Detection Tab Content ===
with tab1:
    st.header("Distraction Detection")
    st.markdown("""
    This module helps monitor attention levels by detecting:
    - Face turning away from the screen
    - Eyes closing for prolonged periods
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        source_option = st.radio("Select video source:", ["Webcam", "Upload Video File"])
        
        if source_option == "Webcam":
            live_placeholder = st.empty()
            start_webcam = st.button("Start Webcam Monitoring")
            stop_webcam = st.button("Stop Webcam Monitoring")
            
            if start_webcam:
                cap = cv2.VideoCapture(0)
                ear_buffer, face_dir_buffer = [], []
                face_turn_start = eye_close_start = None
                in_distraction = False
                last_distraction_time = 0
                
                while cap.isOpened() and not stop_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame from webcam")
                        break
                    
                    processed_frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, distraction_detected, distraction_type = process_frame(
                        frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time
                    )
                    
                    # Convert to RGB for st.image
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    live_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                    
                    # Add small delay to reduce CPU usage
                    time.sleep(0.01)
                
                cap.release()
        
        else:  # Upload Video File
            uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
            
            if uploaded_file is not None:
                # Save uploaded file to temp directory
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                tfile.close()
                
                # Process the uploaded video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    st.error("Error opening video file")
                else:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    video_placeholder = st.empty()
                    
                    # Reset distraction count for new video
                    st.session_state.distraction_count = 0
                    st.session_state.distraction_events = []
                    
                    # Process video
                    ear_buffer, face_dir_buffer = [], []
                    face_turn_start = eye_close_start = None
                    in_distraction = False
                    last_distraction_time = 0
                    frame_count = 0
                    
                    process_button = st.button("Process Video")
                    if process_button:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            # Process every 2nd frame to speed up analysis
                            if frame_count % 2 == 0:
                                processed_frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time, distraction_detected, distraction_type = process_frame(
                                    frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time
                                )
                                
                                # Display processed frame
                                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                                
                                # Update progress
                                progress_percentage = min(frame_count / total_frames, 1.0)
                                progress_bar.progress(progress_percentage)
                            
                            frame_count += 1
                            
                            # Control playback speed
                            time.sleep(1/fps)
                        
                        cap.release()
                        os.unlink(video_path)  # Remove temp file
                        st.success("Video processing complete!")
    
    with col2:
        st.subheader("Distraction Stats")
        st.metric("Total Distractions", st.session_state.distraction_count)
        
        # Display distraction events
        if st.session_state.distraction_events:
            st.subheader("Distraction Events")
            for event in st.session_state.distraction_events:
                st.info(event)
        
        # Add a reset button
        if st.button("Reset Stats"):
            st.session_state.distraction_count = 0
            st.session_state.distraction_events = []
            st.rerun()

# === Speech Processing Tab Content ===
with tab2:
    st.header("Speech Processing")
    st.markdown("Choose an input method to transcribe and analyze speech:")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["🎤 Record from Microphone", "📁 Upload Audio File"]
        )
        
        if input_method == "🎤 Record from Microphone":
            st.markdown("### Microphone Recording")
            
            # Recording controls with improved logic
            recording_col1, recording_col2 = st.columns(2)
            
            with recording_col1:
                if st.button("🎤 Start Recording", type="primary", disabled=st.session_state.is_recording):
                    # Initialize recording
                    st.session_state.is_recording = True
                    st.session_state.live_transcript = ""
                    st.session_state.transcript = ""
                    st.session_state.summary = ""
                    st.session_state.important_points = []
                    st.session_state.recording_start_time = time.time()
                    st.rerun()
            
            with recording_col2:
                if st.button("⏹ Stop Recording", type="secondary", disabled=not st.session_state.is_recording):
                    # Stop recording
                    st.session_state.is_recording = False
                    st.session_state.transcript = st.session_state.live_transcript
                    st.rerun()
            
            # Show recording status and live transcription
            if st.session_state.is_recording:
                st.warning("🔴 **Recording in progress...**")
                # Call the live recording function
                record_and_transcribe_live()
            
            # Show results after recording stops
            elif st.session_state.live_transcript and not st.session_state.is_recording:
                st.success("✅ **Recording completed!**")
                
                # Show final transcript
                st.markdown("#### 📝 Final Transcript:")
                st.text_area("", st.session_state.live_transcript, height=150, disabled=True, key="final_transcript")
                
                # Offer summarization if transcript is long enough
                if len(st.session_state.live_transcript.split()) > 20:
                    if st.button("📊 Generate Summary", type="primary"):
                        with st.spinner("Generating summary and key points..."):
                            summary, key_points = summarize_text(st.session_state.live_transcript)
                            st.session_state.summary = summary
                            st.session_state.important_points = key_points
                            st.rerun()
        
        elif input_method == "📁 Upload Audio File":
            st.markdown("### Audio File Upload")
            
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="Supported formats: WAV, MP3, M4A, FLAC"
            )
            
            if uploaded_file is not None:
                st.audio(uploaded_file, format='audio/wav')
                
                if st.button("📝 Transcribe Audio", type="primary"):
                    with st.spinner("Processing audio file..."):
                        transcript = process_uploaded_audio(uploaded_file)
                        st.session_state.transcript = transcript
                        
                        # Clear previous summary
                        st.session_state.summary = ""
                        st.session_state.important_points = []
                        
                        st.rerun()
    
    with col2:
        st.markdown("### Results")
        
        # Display transcript (works for both recording modes)
        transcript_to_show = st.session_state.get('transcript', '') or st.session_state.get('live_transcript', '')
        
        if transcript_to_show and not st.session_state.get('is_recording', False):
            st.markdown("#### 📝 Transcript:")
            st.text_area("", transcript_to_show, height=150, disabled=True, key="result_transcript")
            
            # Summarization option
            if len(transcript_to_show.split()) > 20:
                if st.button("📊 Generate Summary", key="summary_button"):
                    with st.spinner("Generating summary and key points..."):
                        summary, key_points = summarize_text(transcript_to_show)
                        st.session_state.summary = summary
                        st.session_state.important_points = key_points
                        st.rerun()
        
        # Display summary and key points
        if st.session_state.get('summary'):
            st.markdown("#### 📋 Summary:")
            st.write(st.session_state.summary)
        
        if st.session_state.get('important_points'):
            st.markdown("#### 🔑 Key Points:")
            for i, point in enumerate(st.session_state.important_points, 1):
                st.write(f"**{i}.** {point}")
    
    # Download options
    final_transcript = st.session_state.get('transcript', '') or st.session_state.get('live_transcript', '')
    
    if final_transcript and not st.session_state.get('is_recording', False):
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📄 Download Transcript",
                final_transcript,
                "transcript.txt",
                "text/plain"
            )
        
        with col2:
            if st.session_state.get('summary'):
                st.download_button(
                    "📊 Download Summary", 
                    st.session_state.summary,
                    "summary.txt",
                    "text/plain"
                )
        
        with col3:
            if st.session_state.get('important_points'):
                key_points_text = "\n".join([f"{i}. {point}" for i, point in enumerate(st.session_state.important_points, 1)])
                st.download_button(
                    "🔑 Download Key Points",
                    key_points_text,
                    "key_points.txt", 
                    "text/plain"
                )

# === NEW PARALLEL MODE TAB ===
with tab3:
    st.header("🔄 Parallel Mode - Comprehensive Monitoring")
    st.markdown("""
    **Parallel Mode** runs both distraction detection and speech processing simultaneously:
    - **Left Panel**: Live webcam feed with distraction detection
    - **Right Panel**: Real-time speech transcription
    - **Combined Results**: Comprehensive monitoring of attention and content understanding
    """)
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Parallel Monitoring", 
                    type="primary", 
                    disabled=st.session_state.parallel_active,
                    help="Start both webcam monitoring and speech recognition"):
            try:
                # Reset session counters for new session
                st.session_state.distraction_count = 0
                st.session_state.distraction_events = []
                st.session_state.live_transcript = ""
                
                start_parallel_monitoring()
                st.success("✅ Parallel monitoring started!")
                time.sleep(0.5)  # Small delay for initialization
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start parallel monitoring: {str(e)}")
    
    with col2:
        if st.button("⏸️ Stop Parallel Monitoring", 
                    type="secondary", 
                    disabled=not st.session_state.parallel_active,
                    help="Stop both monitoring systems"):
            try:
                stop_parallel_monitoring()
                st.success("✅ Parallel monitoring stopped!")
                st.rerun()
            except Exception as e:
                st.error(f"Error stopping monitoring: {str(e)}")
    
    with col3:
        if st.button("🔄 Reset All Data", 
                    help="Clear all accumulated data"):
            st.session_state.distraction_count = 0
            st.session_state.distraction_events = []
            st.session_state.live_transcript = ""
            st.session_state.transcript = ""
            st.session_state.summary = ""
            st.session_state.important_points = []
            st.session_state.parallel_results = {
                'current_frame': None,
                'current_transcript': "",
                'distraction_status': "Ready to start...",
                'recording_status': "Ready to start...",
                'distraction_count': 0,
                'distraction_events': []
            }
            st.success("✅ All data reset!")
            st.rerun()
    
    # Show current status
    if st.session_state.parallel_active:
        st.info("🔴 **PARALLEL MONITORING ACTIVE** - Both systems are running simultaneously")
    else:
        st.info("⚪ **PARALLEL MONITORING INACTIVE** - Click 'Start' to begin comprehensive monitoring")
    
    # Main display area
    if st.session_state.parallel_active:
        # Create two columns for parallel display
        video_col, transcript_col = st.columns([1.2, 0.8])
        
        with video_col:
            st.subheader("📹 Live Distraction Detection")
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Display current frame from queue
            try:
                if not st.session_state.video_frame_queue.empty():
                    current_frame = st.session_state.video_frame_queue.get_nowait()
                    video_placeholder.image(current_frame, channels="RGB", use_column_width=True)
                else:
                    video_placeholder.info("📷 Initializing webcam...")
            except queue.Empty:
                video_placeholder.info("📷 Waiting for video feed...")
            
            # Show distraction status
            distraction_status = st.session_state.parallel_results.get('distraction_status', 'Initializing...')
            if "DISTRACTED" in distraction_status:
                status_placeholder.error(distraction_status)
            elif "FOCUSED" in distraction_status:
                status_placeholder.success(distraction_status)
            else:
                status_placeholder.info(distraction_status)
        
        with transcript_col:
            st.subheader("🎤 Live Speech Processing")
            transcript_placeholder = st.empty()
            recording_status_placeholder = st.empty()
            
            # Display current transcript
            current_transcript = st.session_state.parallel_results.get('current_transcript', '')
            if current_transcript:
                transcript_placeholder.text_area(
                    "Live Transcript:",
                    current_transcript,
                    height=300,
                    key=f"parallel_transcript_{len(current_transcript)}"
                )
            else:
                transcript_placeholder.info("🎙️ Waiting for speech...")
            
            # Show recording status
            recording_status = st.session_state.parallel_results.get('recording_status', 'Initializing...')
            if "error" in recording_status.lower() or "❌" in recording_status:
                recording_status_placeholder.error(recording_status)
            elif "transcribing" in recording_status.lower() or "🎙️" in recording_status:
                recording_status_placeholder.success(recording_status)
            else:
                recording_status_placeholder.info(recording_status)
        
        # Auto-refresh the display every 0.5 seconds when active
        time.sleep(0.5)
        st.rerun()
    
    # Results and Statistics Section
    st.markdown("---")
    st.subheader("📊 Session Statistics & Results")
    
    # Create metrics row - Get data from parallel results or session state
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    # Use parallel results if active, otherwise use session state
    current_distraction_count = (st.session_state.parallel_results.get('distraction_count', 0) 
                                if st.session_state.parallel_active 
                                else st.session_state.distraction_count)
    
    with metrics_col1:
        st.metric("🎯 Total Distractions", current_distraction_count)
    
    with metrics_col2:
        current_transcript = (st.session_state.parallel_results.get('current_transcript', '') 
                            if st.session_state.parallel_active 
                            else st.session_state.get('live_transcript', ''))
        transcript_length = len(current_transcript.split())
        st.metric("📝 Words Transcribed", transcript_length)
    
    with metrics_col3:
        session_duration = 0
        if st.session_state.parallel_active and hasattr(st.session_state, 'session_start_time'):
            session_duration = int(time.time() - st.session_state.session_start_time)
        elif not st.session_state.parallel_active and hasattr(st.session_state, 'session_end_time'):
            session_duration = int(st.session_state.session_end_time - st.session_state.session_start_time)
        st.metric("⏱️ Session Duration (s)", session_duration)
    
    with metrics_col4:
        focus_percentage = 100
        if current_distraction_count > 0 and session_duration > 0:
            # Rough estimation: assume each distraction lasts ~3 seconds
            distraction_time = current_distraction_count * 3
            focus_percentage = max(0, int(100 - (distraction_time / session_duration) * 100))
        st.metric("📈 Focus Percentage", f"{focus_percentage}%")
    
    # Detailed Results Section
    results_col1, results_col2 = st.columns(2)
    
    with results_col1:
        st.subheader("🎯 Distraction Events")
        current_distraction_events = (st.session_state.parallel_results.get('distraction_events', []) 
                                    if st.session_state.parallel_active 
                                    else st.session_state.distraction_events)
        
        if current_distraction_events:
            for i, event in enumerate(current_distraction_events, 1):
                st.write(f"**{i}.** {event}")
        else:
            st.info("No distractions detected in this session! 🎉")
    
    with results_col2:
        st.subheader("📝 Current Transcript")
        final_transcript = (st.session_state.parallel_results.get('current_transcript', '') 
                          if st.session_state.parallel_active 
                          else st.session_state.get('live_transcript', ''))
        
        if final_transcript:
            st.text_area("", final_transcript, height=200, disabled=True, key="parallel_final_transcript")
            
            # Offer summarization for long transcripts
            if len(final_transcript.split()) > 20:
                if st.button("📊 Generate Summary & Key Points", key="parallel_summary"):
                    with st.spinner("Analyzing transcript..."):
                        summary, key_points = summarize_text(final_transcript)
                        st.session_state.summary = summary
                        st.session_state.important_points = key_points
                        st.rerun()
        else:
            st.info("No transcript available yet. Start speaking to see transcription here.")
    
    # Summary and Key Points (if generated)
    if st.session_state.get('summary') or st.session_state.get('important_points'):
        st.markdown("---")
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            if st.session_state.get('summary'):
                st.subheader("📋 Content Summary")
                st.write(st.session_state.summary)
        
        with summary_col2:
            if st.session_state.get('important_points'):
                st.subheader("🔑 Key Points Identified")
                for i, point in enumerate(st.session_state.important_points, 1):
                    st.write(f"**{i}.** {point}")
    
    # Download options for parallel mode
    if final_transcript or current_distraction_events:
        st.markdown("---")
        st.subheader("💾 Export Session Data")
        download_col1, download_col2, download_col3, download_col4 = st.columns(4)
        
        with download_col1:
            if final_transcript:
                st.download_button(
                    "📄 Download Transcript",
                    final_transcript,
                    f"parallel_transcript_{int(time.time())}.txt",
                    "text/plain"
                )
        
        with download_col2:
            if current_distraction_events:
                events_text = "\n".join([f"{i}. {event}" for i, event in enumerate(current_distraction_events, 1)])
                st.download_button(
                    "🎯 Download Distraction Log",
                    events_text,
                    f"distraction_log_{int(time.time())}.txt",
                    "text/plain"
                )
        
        with download_col3:
            if st.session_state.get('summary'):
                st.download_button(
                    "📊 Download Summary",
                    st.session_state.summary,
                    f"content_summary_{int(time.time())}.txt",
                    "text/plain"
                )
        
        with download_col4:
            if st.session_state.get('important_points'):
                key_points_text = "\n".join([f"{i}. {point}" for i, point in enumerate(st.session_state.important_points, 1)])
                st.download_button(
                    "🔑 Download Key Points",
                    key_points_text,
                    f"key_points_{int(time.time())}.txt",
                    "text/plain"
                )

# Add a sidebar with app information and settings
with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    This application is designed to help students with learning disabilities stay engaged with their learning materials.
    
    **Features:**
    - **Distraction Detection**: Monitors attention through webcam
    - **Speech Processing**: Transcribes and summarizes spoken content  
    - **Parallel Mode**: Runs both features simultaneously for comprehensive monitoring
    """)
    
    st.header("⚙️ Settings")
    
    # Distraction detection settings
    st.subheader("🎯 Distraction Detection")
    face_turn_threshold = st.slider("Face Turn Threshold (sec)", 1.0, 5.0, float(FACE_TURN_THRESHOLD), 0.5)
    eye_closed_threshold = st.slider("Eye Closed Threshold (sec)", 2.0, 10.0, float(EYE_CLOSED_THRESHOLD), 0.5)
    
    # Speech processing settings
    st.subheader("🎤 Speech Processing")
    auto_summarize = st.checkbox("Auto-summarize long transcriptions", value=True)
    min_words_for_summary = st.slider("Min words for auto-summary", 10, 50, 20, 5)
    
    # Parallel mode settings
    st.subheader("🔄 Parallel Mode")
    update_frequency = st.slider("Display Update Frequency (ms)", 100, 2000, 500, 100)
    st.info("Lower values = more responsive but higher CPU usage")
    
    # System status
    st.markdown("---")
    st.subheader("🖥️ System Status")
    
    if st.session_state.parallel_active:
        st.success("🟢 Parallel Mode: ACTIVE")
        
        # Thread status
        if st.session_state.distraction_thread and st.session_state.distraction_thread.is_alive():
            st.success("📹 Distraction Thread: Running")
        else:
            st.error("📹 Distraction Thread: Stopped")
            
        if st.session_state.speech_thread and st.session_state.speech_thread.is_alive():
            st.success("🎤 Speech Thread: Running")
        else:
            st.error("🎤 Speech Thread: Stopped")
    else:
        st.info("⚪ Parallel Mode: INACTIVE")
    
    # Emergency stop button
    if st.session_state.parallel_active:
        st.markdown("---")
        if st.button("🛑 Emergency Stop", type="secondary", help="Force stop all threads"):
            stop_parallel_monitoring()
            st.error("🛑 Emergency stop activated!")
            st.rerun()
    
    # Performance indicators
    if st.session_state.parallel_active:
        st.markdown("---")
        st.subheader("📊 Performance")
        
        # Queue sizes
        video_queue_size = st.session_state.video_frame_queue.qsize()
        transcription_queue_size = st.session_state.transcription_queue.qsize()
        
        st.metric("📹 Video Queue", f"{video_queue_size}/5")
        st.metric("🎤 Audio Queue", transcription_queue_size)
        
        if video_queue_size >= 4:
            st.warning("⚠️ Video processing may be lagging")
        if transcription_queue_size >= 10:
            st.warning("⚠️ Speech processing may be lagging")
    
    # Add recording indicator when active
    if st.session_state.is_recording:
        st.sidebar.markdown("### 🔴 Recording in progress")
        if st.session_state.recording_start_time:
            duration = time.time() - st.session_state.recording_start_time
            st.sidebar.markdown(f"Duration: {int(duration//60):02d}:{int(duration%60):02d}")
    
    # Add version info at the bottom
    st.markdown("---")
    st.caption("Educational Assistant v2.0 - Parallel Mode")
    st.caption("🔄 Real-time monitoring with multithreading support")