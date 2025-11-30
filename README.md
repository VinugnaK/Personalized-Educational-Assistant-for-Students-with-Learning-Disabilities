# Personalized Educational Assistant for Students with Learning Disabilities

A multimodal **Streamlit application** combining **real-time distraction detection**, **speech processing**, and an integrated **Parallel Monitoring Mode** for evaluating both attention and comprehension simultaneously.

---

## 📌 Overview

This project provides a personalized learning assistant designed for **students with learning disabilities**.  
It leverages:

- **Computer Vision** (MediaPipe + OpenCV)  
- **Speech Recognition**  
- **Natural Language Processing (NLP)**  

to **monitor engagement**, **transcribe spoken content**, and **summarize it into key points**.

The system operates in **three modes**:

1. **Distraction Detection** – Webcam-based attention monitoring  
2. **Speech Processing** – Live or uploaded audio transcription + AI summarization  
3. **Parallel Mode** – Simultaneous video + audio analysis for live sessions  

---

## 🔧 Core Features

### **1. Distraction Detection**
- Detects face turning away  
- Detects eyes closed  
- MediaPipe FaceMesh + OpenCV  
- Supports webcam or uploaded video  
- Logs distraction events with timestamps  

---

### **2. Speech Processing**
- Live microphone transcription  
- Upload audio files (WAV/MP3/M4A/FLAC)  
- Summarization using **BART-Large**  
- Extracts **5 key points**  
- Exportable results  

---

### **3. Parallel Mode**
Runs webcam + speech analysis together:

- Real-time distraction detection  
- Real-time transcription  
- Unified dashboard  
- Session metrics (focus %, words spoken, distractions, duration)  

---

## 🛠️ Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/VinugnaK/Personalized-Educational-Assistant-for-Students-with-Learning-Disabilities
cd Personalized-Educational-Assistant-for-Students-with-Learning-Disabilities
```

### **2. Install Requirements**
```bash
pip install streamlit opencv-python mediapipe transformers speechrecognition numpy pillow sounddevice soundfile matplotlib
```

### **3. Run the Application**
```bash
streamlit run app.py
```

---

## 🧪 Usage

### **Mode 1: Distraction Detection**
- Start webcam  
- Or upload a video  
- System detects eye-closure + face deviation  

### **Mode 2: Speech Processing**
- Record via microphone  
- Or upload audio  
- Generates transcript + summary + key points  

### **Mode 3: Parallel Mode**
- Runs distraction + speech analysis simultaneously  
- Produces combined session report  

---

## 📊 Output & Reports
Exports available in `.txt`:
- Transcript  
- Summary  
- Key Points  
- Distraction Log  
- Session Metrics  

---

## 🧩 Technologies Used

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| Face Tracking | MediaPipe FaceMesh |
| Video Processing | OpenCV |
| Speech Recognition | Google Speech API |
| Summarization | HuggingFace BART-Large |
| Multithreading | Python threading + queue |
| Audio Recording | PyAudio / SoundDevice |

---

## ⚠️ Notes
- First use downloads ~400MB model  
- Webcam must not be used by other applications  
- Speech accuracy depends on noise  
- Parallel mode requires moderate CPU  

---
