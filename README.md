# sentiment___analysis
# SentiAI – Advanced Sentiment & Emotion Intelligence Dashboard  
🧠 A multi-modal sentiment analysis dashboard combining deep-learning (BERT) and lexicon-based (VADER) models — supports text, voice, and video inputs, and features rich visualization & history tracking.

---

## Table of Contents  
- [Features](#-features)  
- [Why SentiAI](#-why-sentiai)  
- [Technology Stack](#-technology-stack)  
- [Getting Started](#-getting-started)  
- [Usage](#-usage)  
- [How It Works](#-how-it-works)  
- [Demo Screenshots](#-demo-screenshots)  
- [Project Structure](#-project-structure)  
- [Contributing](#-contributing)  
- [License](#-license)

---

 Features  
-  Dual-model sentiment engine powered by BERT (transformer-based) + VADER (lexicon-based)  
-  Multi-modal input: ★ Text ★ Voice (speech-to-text) ★ Video (audio extraction + transcription)  
-  Confidence scoring and emotion profiling (joy, sadness, anger, fear, trust, surprise)  
-  Interactive visualizations: sentiment distribution, trend charts  
-  Session history tracking with exportable CSV  
-  Sleek modern dashboard with UI enhancements (glassmorphism, gradient design, responsive layout)

---

Why SentiAI  
Many sentiment tools focus solely on text-based classification. SentiAI elevates this by offering **real-time voice and video support**, combined with dual-model analysis for **improved accuracy** and **user-friendly insights**. Ideal for presentations, social-media monitoring, and AI-driven customer feedback analysis.



  Technology Stack  
| Layer        | Tools / Libraries                                            |
|--------------|-------------------------------------------------------------|
| Backend      | Python, Streamlit                                           |
| NLP Models   | Hugging Face Transformers (BERT), VADER Sentiment Analyzer  |
| Speech/Audio | `speech_recognition`, `pydub`                                |
| Visualizations | Plotly Express, Plotly Graph Objects                    |
| UI/UX        | CSS (glassmorphism, gradients), responsive components       |

---

Getting Started  
### Prerequisites  
- Python 3.8+  
- ffmpeg (for audio/video processing)  
- Microphone & webcam (for voice/video modes)  

### Installation  
```bash
git clone https://github.com/shreyasbp317/sentiment___analysis.git  
cd sentiment___analysis  
pip install -r requirements.txt  
