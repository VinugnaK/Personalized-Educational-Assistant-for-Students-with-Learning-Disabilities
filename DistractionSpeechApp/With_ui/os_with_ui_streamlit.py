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

# Title and description
st.title("Educational Assistant for Students with Learning Disabilities")
st.markdown("""
This application helps students stay focused and understand content through two main features:
1. **Distraction Detection**: Monitors for lack of attention through webcam
2. **Speech Processing**: Transcribes and summarizes speech or audio files
""")

# Create tabs for the two main modules
tab1, tab2 = st.tabs(["Distraction Detection", "Speech Processing"])

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

def process_frame(frame, ear_buffer, face_dir_buffer, face_turn_start, eye_close_start, in_distraction, last_distraction_time):
    """Process a single frame for distraction detection"""
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

# Function for audio recording in a separate thread - completely rewritten
def record_audio():
    """Record audio from microphone and transcribe in real-time"""
    import queue  # Add this import here to make sure it's available
    global audio_queue
    audio_queue = queue.Queue()
    
    # Set recording parameters
    fs = 44100  # Sample rate
    channels = 1  # Mono
    
    # Create a temporary file for the full recording
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    st.session_state.audio_file = temp_audio.name
    temp_audio.close()
    
    # Create a temporary directory for audio chunks
    temp_chunk_dir = tempfile.mkdtemp()
    
    # Provide immediate visual feedback
    st.session_state.is_recording = True
    st.session_state.stop_recording = False
    st.session_state.live_transcript = "Waiting for speech..."
    st.session_state.recording_complete = False
    st.session_state.audio_chunks = []
    
    # Start the recording in a separate thread to keep the UI responsive
    def audio_callback(indata, frames, time_info, status):
        """This function is called for each audio block"""
        if status:
            print(f"Audio callback status: {status}")
        # Add the audio data to the queue
        audio_queue.put(indata.copy())
    
    # Create the wave file with proper parameters
    wf = wave.open(st.session_state.audio_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(2)  # 16-bit audio
    wf.setframerate(fs)
    
    # Process audio from the microphone in real-time
    try:
        # Start the input stream
        with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
            chunk_index = 0
            chunk_start_time = time.time()
            recording_start_time = time.time()
            chunk_buffer = []
            
            # Process audio while recording
            while not st.session_state.stop_recording:
                # Get audio data from the queue with a timeout
                try:
                    audio_block = audio_queue.get(timeout=0.1)
                    wf.writeframes((audio_block * 32767).astype(np.int16).tobytes())
                    chunk_buffer.append(audio_block)
                    
                    # Calculate audio level for visualization
                    audio_level = np.sqrt(np.mean(np.square(audio_block))) * 3000
                    level_percentage = min(100, int(audio_level))
                    
                    # Update the audio level display
                    st.session_state.last_update_time = time.time()
                    st.session_state.level_percentage = level_percentage
                    st.session_state.recording_duration = int(time.time() - recording_start_time)
                    
                    # Process chunks for transcription
                    current_time = time.time()
                    if current_time - chunk_start_time >= 2.0:  # Process every 2 seconds
                        # Save this chunk for transcription
                        chunk_file = os.path.join(temp_chunk_dir, f"chunk_{chunk_index}.wav")
                        with sf.SoundFile(chunk_file, mode='w', samplerate=fs, 
                                         channels=channels, subtype='PCM_16') as chunk_f:
                            for block in chunk_buffer:
                                chunk_f.write(block)
                        
                        st.session_state.audio_chunks.append(chunk_file)
                        
                        # Perform transcription in a separate thread to avoid blocking
                        threading.Thread(target=transcribe_chunk, 
                                         args=(chunk_file, chunk_index)).start()
                        
                        # Reset for next chunk
                        chunk_buffer = []
                        chunk_index += 1
                        chunk_start_time = current_time
                
                except queue.Empty:
                    continue
    
    except Exception as e:
        st.error(f"Error during recording: {e}")
    finally:
        # Close the wave file
        wf.close()
        st.session_state.is_recording = False
        st.session_state.recording_complete = True

def transcribe_chunk(chunk_file, chunk_index):
    """Transcribe a single audio chunk and update the live transcript"""
    try:
        with sr.AudioFile(chunk_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
            # Update the transcript
            if text:
                if st.session_state.live_transcript == "Waiting for speech...":
                    st.session_state.live_transcript = text
                else:
                    st.session_state.live_transcript += " " + text
    except sr.UnknownValueError:
        # No speech detected in this chunk
        pass
    except Exception as e:
        print(f"Error transcribing chunk {chunk_index}: {e}")

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
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Audio Input")
        source_option = st.radio("Select audio source:", ["Microphone", "Upload Audio File"])
        
        if source_option == "Microphone":
            # Record from microphone
            record_status = st.empty()
            
            # Prepare buttons side by side
            col_rec1, col_rec2 = st.columns(2)
            start_button = col_rec1.button("Start Recording")
            stop_button = col_rec2.button("Stop Recording")
            
            # Audio level indicator
            audio_level_placeholder = st.empty()
            
            if start_button and not st.session_state.is_recording:
                # Reset previous results
                st.session_state.transcript = ""
                st.session_state.live_transcript = ""
                st.session_state.summary = ""
                st.session_state.important_points = []
                st.session_state.recording_complete = False
                st.session_state.level_percentage = 0
                st.session_state.recording_duration = 0
                
                # Show immediate feedback before starting thread
                record_status.markdown("<h3 style='color:red'>🔴 STARTING MICROPHONE...</h3>", unsafe_allow_html=True)
                
                # Start recording in a thread
                recording_thread = threading.Thread(target=record_audio)
                recording_thread.daemon = True
                recording_thread.start()
                
                # Give the thread time to initialize
                time.sleep(1)
                st.rerun()
            
            # Show real-time audio level and transcript during recording
            if st.session_state.is_recording:
                # Display recording status
                record_status.markdown("<h3 style='color:red'>🔴 RECORDING ACTIVE - SPEAK NOW</h3>", unsafe_allow_html=True)
                
                # Display audio level with more frequent updates
                if hasattr(st.session_state, 'level_percentage'):
                    level_percentage = st.session_state.level_percentage
                    duration = st.session_state.recording_duration
                    
                    audio_level_placeholder.markdown(f"""
                    <div style="background-color:#f0f0f0; border-radius:5px; padding:10px">
                        <div style="color:red; font-size:16px">Recording: {duration} seconds</div>
                        <div style="background-color:#e1e1e1; border-radius:3px; margin:5px 0">
                            <div style="width:{level_percentage}%; background-color:{'green' if level_percentage > 20 else 'gray'}; 
                                 height:20px; border-radius:3px; transition:width 0.1s ease"></div>
                        </div>
                        <div>Voice detected: {"Yes ✓" if level_percentage > 10 else "No ✗"}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            if stop_button and st.session_state.is_recording:
                st.session_state.stop_recording = True
                record_status.warning("Stopping recording... Please wait...")
                
                # Wait for recording to complete
                time.sleep(2)
                
                # Final full transcription using the complete audio file
                if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                    try:
                        with sr.AudioFile(st.session_state.audio_file) as source:
                            audio_data = recognizer.record(source)
                            text = recognizer.recognize_google(audio_data)
                            st.session_state.transcript = text
                    except Exception as e:
                        # If full transcription fails, use the accumulated live transcript
                        st.session_state.transcript = st.session_state.live_transcript
                    
                    # Display audio player
                    st.audio(st.session_state.audio_file)
                    record_status.success("✅ Recording complete! Audio saved and transcribed.")
                    
                st.rerun()
        
        else:  # Upload Audio File
            uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
            
            if uploaded_file is not None:
                # Save the uploaded file to a temporary location
                audio_temp = tempfile.NamedTemporaryFile(delete=False)
                audio_temp.write(uploaded_file.read())
                audio_path = audio_temp.name
                audio_temp.close()
                
                # Display audio player
                st.audio(uploaded_file)
                
                # Process button
                if st.button("Transcribe Audio"):
                    with st.spinner("Transcribing audio..."):
                        try:
                            # For WAV files
                            if uploaded_file.name.lower().endswith('.wav'):
                                with sr.AudioFile(audio_path) as source:
                                    audio_data = recognizer.record(source)
                                    text = recognizer.recognize_google(audio_data)
                                    st.session_state.transcript = text
                                    st.session_state.live_transcript = text  # Set both for consistent display
                            else:
                                # For MP3 and other formats
                                st.warning("Non-WAV files may have limited support")
                                try:
                                    # Try to use the file directly with speech_recognition
                                    # This might work for some formats depending on system configuration
                                    with sr.AudioFile(audio_path) as source:
                                        audio_data = recognizer.record(source)
                                        text = recognizer.recognize_google(audio_data)
                                        st.session_state.transcript = text
                                        st.session_state.live_transcript = text  # Set both for consistent display
                                except Exception as e:
                                    st.error(f"Could not process this audio format: {e}")
                                    st.session_state.transcript = "Could not process this audio format. Please convert to WAV."
                                    st.session_state.live_transcript = st.session_state.transcript
                                
                        except sr.UnknownValueError:
                            st.error("Could not understand audio")
                        except sr.RequestError as e:
                            st.error(f"Error in API request: {e}")
                        except Exception as e:
                            st.error(f"Error processing audio: {e}")
                    
                    # Set recording complete to true to show summary option
                    st.session_state.recording_complete = True
                    
                    # Clean up temp file
                    os.unlink(audio_path)
    
    with col2:
        st.subheader("Transcription Results")
        
        # Live transcript display area (always visible during recording)
        live_transcript_area = st.empty()
        
        # Improved live transcript display with auto-update
        if st.session_state.is_recording or st.session_state.live_transcript:
            live_transcript_area.markdown(f"""
            <div style="background-color:#f7f7f7; border:1px solid #ddd; border-radius:5px; padding:15px; margin:10px 0">
                <h4 style="color:#333; margin-top:0">Real-time Transcript:</h4>
                <div style="background-color:white; padding:10px; border-radius:3px; min-height:100px">
                    {st.session_state.live_transcript if st.session_state.live_transcript else "Listening..."}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add automatic refresh during recording
            if st.session_state.is_recording:
                st.markdown("""
                <script>
                    setTimeout(function(){
                        window.location.reload();
                    }, 2000);
                </script>
                """, unsafe_allow_html=True)
        
        # Show final transcript after recording is complete
        if st.session_state.transcript:
            with st.expander("Full Transcript", expanded=True):
                st.write(st.session_state.transcript)
            
            # Add separate summarization button - only show after recording is complete
            if len(st.session_state.transcript.split()) > 30 and st.session_state.recording_complete:
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        st.session_state.summary, st.session_state.important_points = summarize_text(st.session_state.transcript)
            
            # Show summarization if available
            if st.session_state.summary:
                with st.expander("Summary", expanded=True):
                    st.write(st.session_state.summary)
                
                if st.session_state.important_points:
                    with st.expander("Key Points", expanded=True):
                        for i, point in enumerate(st.session_state.important_points, 1):
                            st.markdown(f"**{i}.** {point}")
            
            # Add download buttons
            st.download_button(
                label="Download Transcript",
                data=st.session_state.transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
            if st.session_state.summary:
                st.download_button(
                    label="Download Summary",
                    data=st.session_state.summary,
                    file_name="summary.txt",
                    mime="text/plain"
                )

# Add a sidebar with app information and settings
with st.sidebar:
    st.header("About")
    st.info("""
    This application is designed to help students with learning disabilities stay engaged with their learning materials.
    
    The distraction detection module monitors attention levels, while the speech processing module helps in understanding spoken content through transcription and summarization.
    """)
    
    st.header("Settings")
    
    # Distraction detection settings
    st.subheader("Distraction Detection")
    face_turn_threshold = st.slider("Face Turn Threshold (sec)", 1.0, 5.0, float(FACE_TURN_THRESHOLD), 0.5)
    eye_closed_threshold = st.slider("Eye Closed Threshold (sec)", 2.0, 10.0, float(EYE_CLOSED_THRESHOLD), 0.5)
    
    # Speech processing settings
    st.subheader("Speech Processing")
    auto_summarize = st.checkbox("Auto-summarize long transcriptions", value=True)
    
    # Add recording indicator when active
    if st.session_state.is_recording:
        st.sidebar.markdown("### 🔴 Recording in progress")
        st.sidebar.markdown(f"Duration: {time.strftime('%M:%S', time.gmtime(time.time() - time.time()))}")
    
    # Add version info at the bottom
    st.markdown("---")
    st.caption("Educational Assistant v1.1")
