import threading
from threading import Thread, Lock
import time
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import speech_recognition as sr
from transformers import pipeline
from collections import Counter
import re

# === Shared Setup ===
lock = Lock()
terminate_event = threading.Event()  # Shared termination flag

# === Process 1: Distraction Detection ===
def distraction_detection():
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)

    def speak(msg):
        engine.say(msg)
        engine.runAndWait()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=3)
    mp_drawing = mp.solutions.drawing_utils

    FACE_TURN_THRESHOLD = 2
    EYE_CLOSED_THRESHOLD = 4
    EAR_THRESHOLD = 0.25
    FRAME_SMOOTHING_WINDOW = 5
    MIN_FACE_SIZE = 10000

    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    NOSE_TIP = 1

    ear_buffer, face_dir_buffer = [], []
    distraction_count = 0
    face_turn_start = eye_close_start = None
    last_distraction_time = 0
    in_distraction = False

    def calculate_ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def get_largest_face(faces, w, h):
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

    cap = cv2.VideoCapture(0)
    while cap.isOpened() and not terminate_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        distraction_detected = False
        now = time.time()

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

                mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_TESSELATION)

        if distraction_detected and not in_distraction:
            distraction_count += 1
            in_distraction = True
            last_distraction_time = now
            with lock:
                print(f"[!] Distraction #{distraction_count} ({distraction_type})")
            speak("Please stay focused" if distraction_type == "Face turned" else "Don't sleep now!")

        if not distraction_detected and in_distraction:
            if (now - last_distraction_time) > 2:
                in_distraction = False

        cv2.putText(frame, f'Distractions: {distraction_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Focus Monitor', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            terminate_event.set()
            break

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()
    with lock:
        print("📷 Distraction detection ended.")

# === Process 2: Speech Transcription + Summarization ===
def speech_transcribe_and_summarize():
    all_text = []
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    alert_phrases = ["please stay focused", "don't sleep now"]

    def summarize():
        full_text = " ".join(all_text)
        if not full_text:
            with lock:
                print("⚠ No transcription recorded yet.")
            return

        with lock:
            print("\n📝 Summarizing...")

        chunks = [full_text[i:i+1024] for i in range(0, len(full_text), 1024)]
        summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in chunks]
        final_summary = " ".join(summaries)

        with lock:
            print("\n📌 Summary:\n", final_summary)

        words = re.findall(r'\w+', final_summary.lower())
        word_counts = Counter(words)
        top_words = [word for word, _ in word_counts.most_common(5)]
        points = [s.strip() for s in final_summary.split('.') if any(word in s.lower() for word in top_words)]

        with lock:
            print("\n🔑 Important Points:\n", " | ".join(points))

    def command_listener():
        while not terminate_event.is_set():
            try:
                cmd = input()
                if cmd.strip().lower() == "sum":
                    summarize()
                    terminate_event.set()
                    break
            except EOFError:
                break

    threading.Thread(target=command_listener, daemon=True).start()

    with lock:
        print("\n🎙 Speak Now (Type 'sum' to summarize and stop)...")

    try:
        while not terminate_event.is_set():
            with sr.Microphone() as source:
                recognizer = sr.Recognizer()
                recognizer.adjust_for_ambient_noise(source, duration=1)
                with lock:
                    print("🕒 Listening...")
                try:
                    audio = recognizer.listen(source, timeout=5)
                    text = recognizer.recognize_google(audio).strip().lower()

                    # Filter out alert phrases
                    if not any(alert in text for alert in alert_phrases):
                        all_text.append(text)
                        with lock:
                            print(f"🗣 Transcribed Text: {text}")
                    else:
                        with lock:
                            print(f"🚫 Ignored alert phrase: {text}")

                except sr.WaitTimeoutError:
                    with lock:
                        print("⌛ Timeout: No speech detected.")
                except sr.UnknownValueError:
                    with lock:
                        print("❓ Could not understand audio.")
                except sr.RequestError as e:
                    with lock:
                        print(f"⚠ Could not request results; {e}")
    except KeyboardInterrupt:
        with lock:
            print("\n🛑 Speech input stopped.")

    with lock:
        print("🎤 Speech transcription ended.")

# === Main Controller ===
if __name__ == "__main__":
    print("🚀 Starting both modules in parallel...\n")

    t1 = Thread(target=distraction_detection)
    t2 = Thread(target=speech_transcribe_and_summarize)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("\n✅ Program ended after summarization.")
