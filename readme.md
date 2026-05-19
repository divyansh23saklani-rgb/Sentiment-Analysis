# AMIE: Ape for Machine Intelligence and Emotion 🦍🧠

**AMIE** is a multifaceted AI framework built to combine machine intelligence with emotional processing. Originally conceived as a basic sentiment analysis tool, this project has evolved into a comprehensive suite of AI modules designed to analyze text, voice, faces, and emotional trauma, with a specific focus on therapeutic applications such as assisting Alzheimer's patients.

---

## 🌟 Key Features & Modules

### 1. 🧠 Sentiment & Emotion Analysis (`simple_analysis.py`)
A robust text analysis engine that utilizes a dual-model approach:
*   **VADER (NLTK):** For rapid, rule-based polarity scoring.
*   **DistilBERT (Hugging Face):** For deep contextual sentiment classification.
*   *Usage:* Run `python simple_analysis.py` for a real-time text analysis loop.

### 2. 🧓 Alzheimer's Patient MCP (`AlzheimerPatient_mcp/`)
A therapeutic module designed to parse patient statements and memories (via CSV/JSON). 
*   **Trauma & Grief Detection:** Analyzes input for emotional significance, temporal context, and grief indicators.
*   **Care Recommendations:** Automatically generates caregiver advice based on the patient's emotional state.
*   *Key files:* `json_based_analysis.py`, `csv_trauma_analysis.py`

### 3. 🗣️ Voice Cloning & Synthesis (`voice_cloning/`)
A suite of tools for real-time and offline voice cloning. 
*   Allows for generating synthesized speech based on sample audio (useful for preserving voices or creating comforting audio for memory patients).
*   *Key files:* `main_clone.py`, `rt_clone.py`, `fixed_mainClone.py`

### 4. 👤 Face Recognition (`face_recognition/`)
Computer vision tools to recognize and identify individuals from images/video streams, which can be applied to assist patients with facial recall.

### 5. 🎵 Song Recognition API (`flask_songRecAPI/`)
A Flask-based backend server capable of taking audio recordings and identifying the song.
*   Can be integrated into music therapy pipelines to help trigger positive memories for patients.
*   *Key files:* `songRecAPI.py`, `song_recongizer.py`

---

## 🛠️ Tech Stack
*   **Core:** Python 3
*   **Machine Learning/AI:** Scikit-learn, Transformers (Hugging Face), NLTK (VADER)
*   **Web Framework:** Flask
*   **Data Processing:** Pandas, NumPy
*   **Environment:** Jupyter Notebook, Ngrok (for local API tunneling)

---

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/divyansh23saklani-rgb/Sentiment-Analysis.git
   cd Sentiment-Analysis
   ```

2. **Install Dependencies**
   ```bash
   pip install -r req.txt
   ```
   *(Note: Some modules like voice cloning and face recognition may require additional system-level dependencies like PyTorch or OpenCV).*

3. **Running the Core Sentiment Analyzer**
   ```bash
   python simple_analysis.py
   ```

4. **Running the Song API**
   ```bash
   cd flask_songRecAPI
   python songRecAPI.py
   ```

---
*Note: This project was built as a rapid prototype (Hackathon style). Some features remain experimental.*
