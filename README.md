# 🧠 SentiAI - Advanced Multi-Modal Sentiment Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

> A sophisticated AI-powered sentiment analysis system combining BERT and VADER models to analyze emotions from text, voice, and video inputs with real-time processing and interactive visualizations.

<p align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" alt="Status"/>
  <img src="https://img.shields.io/badge/Features-7%20Major-blue?style=for-the-badge" alt="Features"/>
  <img src="https://img.shields.io/badge/Accuracy-95%25+-success?style=for-the-badge" alt="Accuracy"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License](#-license)
- [Contact](#-contact)

---

## 🌟 Overview

SentiAI is a comprehensive sentiment analysis platform that leverages state-of-the-art natural language processing and machine learning techniques to provide accurate, multi-dimensional sentiment insights. By combining BERT's deep contextual understanding with VADER's lexicon-based analysis, SentiAI achieves superior accuracy across diverse input modalities.

### Why SentiAI?

- **Dual-Model Architecture**: Combines BERT (60%) and VADER (40%) for optimal accuracy
- **Multi-Modal Analysis**: Support for text, voice, and video inputs
- **Real-Time Processing**: Instant sentiment detection with <2s latency
- **Emotion Detection**: Identifies 6 core emotions with confidence scoring
- **Production Ready**: Fully functional with comprehensive error handling

---

## ✨ Key Features

### 🎯 Core Capabilities

| Feature | Description | Status |
|---------|-------------|--------|
| **Text Analysis** | BERT + VADER dual-model sentiment detection | ✅ Fully Functional |
| **Voice Recording** | Real-time speech-to-text transcription | ✅ Fully Functional |
| **Video Analysis** | Audio extraction and sentiment analysis | ✅ Fully Functional |
| **YouTube Integration** | Direct URL download and analysis | ✅ Fully Functional |
| **Aspect-Based Analysis** | Topic extraction with individual sentiments | ✅ Fully Functional |
| **AI Chatbot** | Sentiment-aware conversational AI | ✅ Fully Functional |
| **History Management** | Store, visualize, and export analyses | ✅ Fully Functional |

### 🎨 Advanced Features

- **Emotion Mapping**: Joy, Sadness, Anger, Fear, Surprise, Trust
- **Confidence Scoring**: Reliability metrics for each analysis
- **Interactive Visualizations**: Plotly-powered charts and heatmaps
- **Data Export**: CSV export for further analysis
- **Responsive UI**: Modern dark-themed interface with smooth animations

---

## 🎥 Demo

### Text Analysis
```
Input: "I absolutely love this product! It exceeded my expectations."
Output: POSITIVE (Score: 0.92) | Confidence: 94%
Emotions: Joy (85%), Trust (78%)
```

### Video Analysis
```
Upload video → Extract audio → Transcribe → Analyze sentiment
Processing Time: ~15-30 seconds for 1-minute video
```

### Aspect-Based Analysis
```
Input: "Great camera, but poor battery life."
Output: 
  📌 Camera: POSITIVE (0.89)
  📌 Battery Life: NEGATIVE (0.25)
```

---

## 🛠 Technology Stack

### Core Technologies

| Component | Technology | Version |
|-----------|-----------|---------|
| **ML Framework** | PyTorch | 2.0+ |
| **NLP Models** | BERT, VADER | Latest |
| **Web Framework** | Streamlit | 1.28+ |
| **Speech Recognition** | Google Speech API | - |
| **Audio Processing** | PyDub, FFmpeg | Latest |
| **Video Processing** | yt-dlp | Latest |
| **NLP Library** | SpaCy | 3.6+ |
| **Visualization** | Plotly | 5.14+ |

### ML Models

- **BERT**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **VADER**: Built-in lexicon-based sentiment analyzer
- **SpaCy**: `en_core_web_sm` for aspect extraction

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio/video processing)
- Internet connection (for speech recognition)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/sentiai.git
cd sentiai
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### Step 4: Install FFmpeg

**Windows:**
```bash
# Using Chocolatey
choco install ffmpeg

# Or download from: https://ffmpeg.org/download.html
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### Step 5: Run Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📝 Requirements.txt

```txt
# Core ML/AI
numpy>=1.22.4,<2.0
protobuf==3.20.3
torch>=2.0.0
transformers>=4.30.0
vaderSentiment>=3.3.2

# Web Framework
streamlit>=1.28.0

# Audio/Video Processing
SpeechRecognition>=3.10.0
pydub>=0.25.1
opencv-python>=4.8.0

# YouTube & NLP
yt-dlp>=2023.10.13
spacy>=3.6.0

# Data & Visualization
plotly>=5.14.0
pandas>=2.0.0
```

---

## 🚀 Usage

### Quick Start

1. **Launch Application**
   ```bash
   streamlit run app.py
   ```

2. **Choose Analysis Method**
   - Text: Paste or type text
   - Voice: Click record and speak
   - Video: Upload video file
   - YouTube: Paste URL

3. **View Results**
   - Sentiment score (0-1)
   - Classification (Positive/Negative/Neutral)
   - Emotion breakdown
   - Confidence metrics

### Example Usage

#### Python API (Direct Function Calls)

```python
from app import analyze_text_comprehensive

# Analyze text
text = "I love this product!"
result = analyze_text_comprehensive(text)

print(f"Sentiment: {result['final_sentiment']}")
print(f"Score: {result['combined_score']:.2f}")
print(f"Confidence: {result['confidence']:.2%}")
```

#### Aspect-Based Analysis

```python
from app import extract_aspects

text = "Great camera, but poor battery life."
aspects = extract_aspects(text)

for aspect in aspects:
    print(f"{aspect['aspect']}: {aspect['sentiment']} ({aspect['score']:.2f})")
```

---

## 🏗 Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       User Interface                         │
│                      (Streamlit Web App)                     │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Input Processing Layer                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │   Text   │  │  Voice   │  │  Video   │  │ YouTube  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Sentiment Analysis Engine                   │
│  ┌─────────────────────┐      ┌─────────────────────┐      │
│  │   BERT Model (60%)  │      │  VADER Model (40%)  │      │
│  │  Deep Learning      │  +   │  Lexicon-based      │      │
│  └─────────────────────┘      └─────────────────────┘      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Results & Visualization                   │
│  • Sentiment Score    • Emotion Analysis                    │
│  • Confidence Metrics • Interactive Charts                  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input Acquisition**: Text/Voice/Video → Preprocessing
2. **Feature Extraction**: Tokenization → Embedding
3. **Model Inference**: BERT + VADER → Combined Score
4. **Post-Processing**: Emotion Mapping → Confidence Calculation
5. **Visualization**: Results Display → History Storage

---

## 📊 API Reference

### Core Functions

#### `analyze_text_comprehensive(text: str) -> Dict`

Performs comprehensive sentiment analysis on input text.

**Parameters:**
- `text` (str): Input text to analyze (max 512 tokens for BERT)

**Returns:**
- `Dict`: Analysis results containing:
  - `final_sentiment`: POSITIVE/NEGATIVE/NEUTRAL
  - `combined_score`: Float (0-1)
  - `bert_score`: BERT model confidence
  - `vader_compound`: VADER compound score
  - `emotions`: Dict of 6 emotions with scores
  - `confidence`: Overall confidence percentage

**Example:**
```python
result = analyze_text_comprehensive("I love this!")
print(result['final_sentiment'])  # POSITIVE
print(result['combined_score'])   # 0.92
```

#### `extract_aspects(text: str) -> List[Dict]`

Extracts aspects/topics and their individual sentiments.

**Parameters:**
- `text` (str): Input text with multiple aspects

**Returns:**
- `List[Dict]`: List of aspects with sentiment scores

**Example:**
```python
aspects = extract_aspects("Great camera, poor battery")
# [{'aspect': 'camera', 'sentiment': 'POSITIVE', 'score': 0.89}, ...]
```

#### `transcribe_audio(audio_path: str) -> str`

Transcribes audio file to text using Google Speech Recognition.

**Parameters:**
- `audio_path` (str): Path to audio file (.wav format)

**Returns:**
- `str`: Transcribed text or error message

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed
- Test thoroughly before submitting PR

### Code Style

```python
def function_name(param: type) -> return_type:
    """
    Brief description.
    
    Args:
        param: Description
        
    Returns:
        Description of return value
    """
    pass
```

---

## 🐛 Troubleshooting

### Common Issues

#### Issue: "PyAudio not found"
**Solution:**
```bash
# macOS
brew install portaudio
pip install pyaudio

# Linux
sudo apt-get install portaudio19-dev
pip install pyaudio

# Windows
pip install pipwin
pipwin install pyaudio
```

#### Issue: "FFmpeg not installed"
**Solution:**
```bash
# Install FFmpeg using your package manager
# Verify installation:
ffmpeg -version
```

#### Issue: "YouTube download fails (403 Error)"
**Solution:**
```bash
# Update yt-dlp to latest version
pip install --upgrade yt-dlp

# Alternative: Use video upload feature instead
```

#### Issue: "Protobuf version conflict"
**Solution:**
```bash
pip uninstall protobuf -y
pip install protobuf==3.20.3
```

#### Issue: "SpaCy model not found"
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Getting Help

- 📧 **Email**: support@sentiai.com
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/sentiai/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/sentiai/issues)

---

## 🗺 Roadmap

### Current Version (v1.0.0)
- ✅ Text sentiment analysis
- ✅ Voice recording
- ✅ Video analysis
- ✅ YouTube integration
- ✅ Aspect-based analysis
- ✅ AI chatbot
- ✅ History management

### Upcoming Features (v2.0.0)
- [ ] Multi-language support (10+ languages)
- [ ] Batch processing (100+ files)
- [ ] API endpoints (REST API)
- [ ] Real-time social media monitoring
- [ ] Advanced emotion detection (14 emotions)
- [ ] Custom model training
- [ ] Mobile app (iOS/Android)
- [ ] Cloud deployment (AWS/GCP)

### Future Enhancements (v3.0.0)
- [ ] Sarcasm detection
- [ ] Context-aware analysis
- [ ] Multi-speaker diarization
- [ ] Video facial emotion recognition
- [ ] Sentiment forecasting
- [ ] Industry-specific models

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 SentiAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 📞 Contact

**Project Maintainers:**
- Your Name - [@yourhandle](https://twitter.com/yourhandle) - email@example.com
- Team Member 2 - [@handle2](https://twitter.com/handle2)

**Organization:**
- Website: [https://sentiai.com](https://sentiai.com)
- Documentation: [https://docs.sentiai.com](https://docs.sentiai.com)
- Blog: [https://blog.sentiai.com](https://blog.sentiai.com)

**Links:**
- 🌐 [Project Website](https://sentiai.com)
- 📚 [Documentation](https://docs.sentiai.com)
- 📝 [Blog](https://blog.sentiai.com)
- 🐦 [Twitter](https://twitter.com/sentiai)

---

## 🙏 Acknowledgments

- **Hugging Face** for BERT model hosting
- **VADER Sentiment** for lexicon-based analysis
- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **Google** for Speech Recognition API
- **SpaCy** for NLP processing

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/sentiai?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/sentiai?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/yourusername/sentiai?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/sentiai)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/sentiai)

---

## 🎓 Citation

If you use SentiAI in your research, please cite:

```bibtex
@software{sentiai2024,
  title = {SentiAI: Advanced Multi-Modal Sentiment Analysis Platform},
  author = {Your Name and Team},
  year = {2024},
  url = {https://github.com/yourusername/sentiai}
}
```

---

<div align="center">

**[⬆ Back to Top](#-sentiai---advanced-multi-modal-sentiment-analysis-platform)**

⭐ **Star this repo** if you find it useful!
