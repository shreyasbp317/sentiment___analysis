import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
from pydub import AudioSegment
import os
import time
import pandas as pd
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import subprocess

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="SentiAI - Advanced Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS for Advanced UI/UX
# ----------------------------
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] .css-1d391kg {
        color: #f1f5f9;
    }

    /* Header Styles */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        text-align: center;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Feature Badges */
    .feature-badges {
        display: flex;
        justify-content: center;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 2rem 0;
    }

    .badge {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid #6366f1;
        color: #6366f1;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #334155;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #6366f1;
        transform: translateY(-2px);
        box-shadow: 0 15px 50px rgba(99, 102, 241, 0.2);
    }

    .sentiment-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }

    .sentiment-score {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .sentiment-label {
        font-size: 1.3rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(79, 172, 254, 0.3);
    }

    /* Input Fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid #334155;
        border-radius: 8px;
        color: #f1f5f9;
        padding: 12px;
    }

    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 23, 42, 0.5);
        border-radius: 8px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #94a3b8;
        border-radius: 6px;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: #6366f1;
        color: white;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
    }

    /* Dataframe */
    .dataframe {
        background: #1e293b;
        border-radius: 8px;
        border: 1px solid #334155;
    }

    /* Alert Boxes */
    .stAlert {
        background: rgba(99, 102, 241, 0.1);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
    }

    /* Emotion Card */
    .emotion-card {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #334155;
        margin: 0.5rem;
    }

    .emotion-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }

    .emotion-name {
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }

    .emotion-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #06b6d4;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.5);
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 2rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #6366f1;
        background: rgba(99, 102, 241, 0.05);
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #06b6d4;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Enhanced Model Loading with BERT + VADER
# ----------------------------
@st.cache_resource
def load_models():
    """Load both BERT and VADER models for accurate sentiment analysis"""
    try:
        # Load BERT model for deep learning analysis
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        bert_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        # Load VADER for lexicon-based analysis
        vader = SentimentIntensityAnalyzer()

        return bert_pipeline, vader
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None


# Load models
bert_analyzer, vader_analyzer = load_models()


# ----------------------------
# Enhanced Analysis Functions
# ----------------------------
def analyze_text_comprehensive(text: str) -> Dict:
    """Comprehensive sentiment analysis using both BERT and VADER"""
    if not text or not text.strip():
        return None

    try:
        # BERT Analysis
        bert_result = bert_analyzer(text[:512])[0]  # BERT has token limit
        bert_label = bert_result['label']
        bert_score = float(bert_result['score'])

        # Convert BERT rating to sentiment
        if '5 stars' in bert_label or '4 stars' in bert_label:
            bert_sentiment = 'POSITIVE'
            bert_normalized = 0.7 + (bert_score * 0.3)
        elif '3 stars' in bert_label:
            bert_sentiment = 'NEUTRAL'
            bert_normalized = 0.4 + (bert_score * 0.2)
        else:
            bert_sentiment = 'NEGATIVE'
            bert_normalized = bert_score * 0.4

        # VADER Analysis
        vader_scores = vader_analyzer.polarity_scores(text)
        vader_compound = vader_scores['compound']

        # Determine VADER sentiment
        if vader_compound >= 0.05:
            vader_sentiment = 'POSITIVE'
        elif vader_compound <= -0.05:
            vader_sentiment = 'NEGATIVE'
        else:
            vader_sentiment = 'NEUTRAL'

        # Normalize VADER score to 0-1
        vader_normalized = (vader_compound + 1) / 2

        # Combined score (weighted average)
        combined_score = (bert_normalized * 0.6) + (vader_normalized * 0.4)

        # Determine final sentiment
        if combined_score >= 0.6:
            final_sentiment = 'POSITIVE'
        elif combined_score <= 0.4:
            final_sentiment = 'NEGATIVE'
        else:
            final_sentiment = 'NEUTRAL'

        # Emotion analysis based on VADER components
        emotions = {
            'joy': max(0, vader_scores['pos'] * 100),
            'sadness': max(0, vader_scores['neg'] * 100),
            'anger': max(0, (vader_scores['neg'] * 0.7) * 100),
            'fear': max(0, (vader_scores['neg'] * 0.3) * 100),
            'surprise': abs(vader_scores['neu'] * 50),
            'trust': max(0, vader_scores['pos'] * 80)
        }

        # Confidence calculation
        confidence = (bert_score + abs(vader_compound)) / 2

        return {
            'text': text,
            'final_sentiment': final_sentiment,
            'combined_score': combined_score,
            'bert_sentiment': bert_sentiment,
            'bert_score': bert_score,
            'bert_label': bert_label,
            'vader_sentiment': vader_sentiment,
            'vader_compound': vader_compound,
            'vader_pos': vader_scores['pos'],
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'emotions': emotions,
            'confidence': confidence,
            'word_count': len(text.split()),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


def recognize_speech() -> str:
    """Enhanced speech recognition with error handling"""
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Listening... Speak clearly!")
            r.adjust_for_ambient_noise(source, duration=1)
            audio = r.listen(source, timeout=10, phrase_time_limit=15)

        text = r.recognize_google(audio)
        return text
    except sr.WaitTimeoutError:
        return "⚠️ Timeout: No speech detected"
    except sr.UnknownValueError:
        return "❌ Could not understand audio"
    except sr.RequestError as e:
        return f"⚠️ Service error: {str(e)}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def extract_audio_from_video(video_file_path: str) -> str:
    """Extract audio from video with robust error handling"""
    try:
        audio_path = "temp_audio.wav"  # Fixed: proper filename

        # Verify input file exists
        if not os.path.exists(video_file_path):
            st.error(f"❌ Video file not found: {video_file_path}")
            return None

        file_size = os.path.getsize(video_file_path)
        st.info(f"📹 Video file size: {file_size / (1024 * 1024):.2f} MB")

        # Load video and extract audio
        try:
            audio = AudioSegment.from_file(video_file_path)
        except Exception as e:
            st.error(f"❌ Could not load video. Error: {str(e)}")
            st.warning("💡 Make sure FFmpeg is installed and in your PATH")
            return None

        # Optimize audio for speech recognition
        audio = audio.set_channels(1)  # Convert to mono
        audio = audio.set_frame_rate(16000)  # Optimal sample rate for speech

        # Export audio
        audio.export(audio_path, format="wav")

        # Verify audio file
        if os.path.exists(audio_path):
            audio_size = os.path.getsize(audio_path)
            st.success(f"✅ Audio extracted ({audio_size / 1024:.2f} KB)")
            return audio_path
        else:
            st.error("❌ Audio file was not created")
            return None

    except Exception as e:
        st.error(f"❌ Audio extraction failed: {str(e)}")
        return None


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio with detailed error handling"""
    r = sr.Recognizer()

    try:
        # Verify file exists
        if not os.path.exists(audio_path):
            return "❌ Error: Audio file not found"

        # Check file size
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            return "❌ Error: Audio file is empty (0 bytes)"

        st.info(f"🎵 Processing audio file ({file_size / 1024:.2f} KB)")

        # Load audio file
        with sr.AudioFile(audio_path) as source:
            # Adjust for ambient noise
            st.info("🔊 Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=0.5)

            # Record audio data
            st.info("📼 Reading audio data...")
            audio_data = r.record(source)

            # Recognize speech
            st.info("🌐 Transcribing with Google Speech Recognition...")
            text = r.recognize_google(audio_data, language='en-US')

            if text and len(text.strip()) > 0:
                return text
            else:
                return "❌ No speech detected in audio"

    except sr.UnknownValueError:
        return "❌ Could not understand audio. Possible reasons:\n• Audio quality is poor\n• No clear speech detected\n• Background noise is too loud"
    except sr.RequestError as e:
        return f"⚠️ Google Speech API error: {str(e)}\n• Check your internet connection\n• API might be temporarily unavailable"
    except ValueError as e:
        return f"❌ Invalid audio format: {str(e)}"
    except Exception as e:
        return f"❌ Transcription error: {str(e)}"


# ----------------------------
# Session State Initialization
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "current_analysis" not in st.session_state:
    st.session_state.current_analysis = None


# ----------------------------
# Helper Functions
# ----------------------------
def display_analysis_results(result: Dict):
    """Display comprehensive analysis results with beautiful UI"""
    if not result:
        return

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    # Main sentiment card
    sentiment_emoji = {
        'POSITIVE': '😊',
        'NEGATIVE': '😞',
        'NEUTRAL': '😐'
    }

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="sentiment-card">
            <div class="sentiment-score">{result['combined_score']:.2f}</div>
            <div class="sentiment-label">{sentiment_emoji.get(result['final_sentiment'], '😐')} {result['final_sentiment']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("BERT Score", f"{result['bert_score']:.2f}", result['bert_sentiment'])
        st.metric("Word Count", result['word_count'])

    with col3:
        st.metric("VADER Score", f"{result['vader_compound']:.2f}", result['vader_sentiment'])
        st.metric("Confidence", f"{result['confidence']:.2%}")

    # Emotion Analysis
    st.markdown("### 😊 Emotion Analysis")

    emotion_cols = st.columns(6)
    emotion_icons = ['😄', '😢', '😠', '😨', '😲', '🤝']
    emotion_names = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Trust']

    for idx, (col, icon, name, key) in enumerate(
            zip(emotion_cols, emotion_icons, emotion_names, result['emotions'].keys())):
        with col:
            value = result['emotions'][key]
            st.markdown(f"""
            <div class="emotion-card">
                <div class="emotion-icon">{icon}</div>
                <div class="emotion-name">{name}</div>
                <div class="emotion-value">{value:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(value / 100)

    # Detailed Scores
    with st.expander("🔍 Detailed Analysis Breakdown"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("*BERT Analysis*")
            st.write(f"- Label: {result['bert_label']}")
            st.write(f"- Sentiment: {result['bert_sentiment']}")
            st.write(f"- Confidence: {result['bert_score']:.2%}")

        with col2:
            st.markdown("*VADER Analysis*")
            st.write(f"- Compound: {result['vader_compound']:.3f}")
            st.write(f"- Positive: {result['vader_pos']:.2%}")
            st.write(f"- Negative: {result['vader_neg']:.2%}")
            st.write(f"- Neutral: {result['vader_neu']:.2%}")


def save_to_history(result: Dict):
    """Save analysis to history"""
    if result:
        st.session_state.history.append(result)


def create_visualizations(df: pd.DataFrame):
    """Create advanced visualizations using Plotly"""

    # Sentiment Distribution
    fig1 = px.pie(
        df['final_sentiment'].value_counts().reset_index(),
        values='count',
        names='final_sentiment',
        title='Sentiment Distribution',
        color_discrete_map={
            'POSITIVE': '#10b981',
            'NEGATIVE': '#ef4444',
            'NEUTRAL': '#f59e0b'
        },
        hole=0.4
    )
    fig1.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Score Trends
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=df['combined_score'],
        mode='lines+markers',
        name='Combined Score',
        line=dict(color='#4facfe', width=3),
        marker=dict(size=8)
    ))
    fig2.add_trace(go.Scatter(
        y=df['bert_score'],
        mode='lines+markers',
        name='BERT Score',
        line=dict(color='#667eea', width=2),
        marker=dict(size=6)
    ))
    fig2.add_trace(go.Scatter(
        y=df['vader_compound'].apply(lambda x: (x + 1) / 2),
        mode='lines+markers',
        name='VADER Score',
        line=dict(color='#764ba2', width=2),
        marker=dict(size=6)
    ))
    fig2.update_layout(
        title='Sentiment Scores Over Time',
        xaxis_title='Analysis #',
        yaxis_title='Score',
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x unified'
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Emotion Heatmap
    emotion_data = []
    for idx, row in df.iterrows():
        emotions = row['emotions']
        emotion_data.append(list(emotions.values()))

    if emotion_data:
        fig3 = go.Figure(data=go.Heatmap(
            z=emotion_data,
            x=list(df.iloc[0]['emotions'].keys()),
            y=list(range(len(emotion_data))),
            colorscale='Viridis',
            text=[[f"{val:.1f}%" for val in row] for row in emotion_data],
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig3.update_layout(
            title='Emotion Intensity Heatmap',
            xaxis_title='Emotions',
            yaxis_title='Analysis #',
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig3, use_container_width=True)


# ----------------------------
# Sidebar Navigation
# ----------------------------
with st.sidebar:
    st.markdown("### 🧠 SentiAI Dashboard")
    st.markdown("*Advanced Sentiment Analysis Platform*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Analyzer", "📚 History", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    total_analyses = len(st.session_state.history)
    st.metric("Total Analyses", total_analyses)

    if total_analyses > 0:
        avg_score = np.mean([h['combined_score'] for h in st.session_state.history])
        st.metric("Avg Score", f"{avg_score:.2f}")

        sentiments = [h['final_sentiment'] for h in st.session_state.history]
        most_common = max(set(sentiments), key=sentiments.count) if sentiments else 'N/A'
        st.metric("Most Common", most_common)

    st.markdown("---")

    # System Check Button
    if st.button("🧪 Test FFmpeg"):
        st.markdown("### 🔍 System Check")

        # Test FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                    capture_output=True,
                                    text=True,
                                    timeout=5)
            if result.returncode == 0:
                st.success("✅ FFmpeg is installed")
                version_line = result.stdout.split('\n')[0]
                st.caption(f"{version_line[:50]}...")
            else:
                st.error("❌ FFmpeg error")
        except FileNotFoundError:
            st.error("❌ FFmpeg NOT installed")
            st.info("Install: https://ffmpeg.org")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    st.markdown("---")
    st.markdown("### 🎓 Project Info")
    st.markdown("*Final Year Project*")
    st.markdown("*College:* Your College Name")
    st.markdown("*Team:* Your Team Name")

    st.markdown("---")
    st.caption("Built with 🧠 BERT + VADER • Streamlit • Transformers")

# ----------------------------
# Main Pages
# ----------------------------

if page == "🏠 Home":
    st.markdown('<div class="hero-title">🧠 SentiAI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Advanced Multi-Modal Sentiment Analysis using BERT and VADER Models</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-badges">
        <span class="badge">🤖 BERT Integration</span>
        <span class="badge">📊 VADER Analysis</span>
        <span class="badge">🎥 Video Processing</span>
        <span class="badge">🎤 Voice Recognition</span>
        <span class="badge">💾 Analysis Storage</span>
        <span class="badge">📈 Advanced Visualizations</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📝 Text Analysis
        Analyze any text input using dual-model approach for maximum accuracy
        """)

    with col2:
        st.markdown("""
        ### 🎤 Voice Analysis
        Real-time speech-to-text transcription with sentiment detection
        """)

    with col3:
        st.markdown("""
        ### 🎥 Video Analysis
        Extract audio from videos and analyze emotional content
        """)

    st.markdown("---")

    st.markdown("### 🎯 Key Features")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:
        st.markdown("""
        ✅ *Dual-Model Analysis*: Combines BERT and VADER for accuracy  
        ✅ *Multi-Modal Input*: Text, Voice, and Video support  
        ✅ *Emotion Detection*: 6 core emotions analyzed  
        ✅ *Real-time Processing*: Instant results with progress tracking  
        """)

    with feature_col2:
        st.markdown("""
        ✅ *History Management*: Store and review all analyses  
        ✅ *Advanced Visualizations*: Interactive Plotly charts  
        ✅ *Export Functionality*: Download results as CSV  
        ✅ *Confidence Scoring*: Reliability metrics for each analysis  
        """)

    st.markdown("---")

    st.info("👉 Navigate to *Analyzer* to start analyzing content!")

elif page == "🔍 Analyzer":
    st.markdown("## 🔍 Advanced Sentiment Analyzer")
    st.markdown("Choose your input method and get comprehensive sentiment analysis with emotion detection.")

    tab1, tab2, tab3 = st.tabs(["📝 Text Input", "🎤 Voice Recording", "🎥 Video Upload"])

    # TEXT ANALYSIS TAB
    with tab1:
        st.markdown("### Enter Text for Analysis")
        text_input = st.text_area(
            "Paste or type your text here",
            placeholder="Enter the text you want to analyze for sentiment...",
            height=200,
            help="The analysis supports up to 512 tokens for BERT model"
        )

        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            analyze_btn = st.button("🚀 Analyze Text", use_container_width=True)
        with col2:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)
        with col3:
            example_btn = st.button("💡 Example", use_container_width=True)

        if example_btn:
            text_input = "I absolutely love this product! It exceeded all my expectations and the customer service was outstanding."
            st.rerun()

        if clear_btn:
            text_input = ""
            st.rerun()

        if analyze_btn:
            if not text_input.strip():
                st.warning("⚠️ Please enter some text to analyze")
            else:
                with st.spinner("🔄 Analyzing with BERT and VADER models..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    result = analyze_text_comprehensive(text_input)

                    if result:
                        st.success("✅ Analysis Complete!")
                        display_analysis_results(result)
                        save_to_history(result)

    # VOICE ANALYSIS TAB
    with tab2:
        st.markdown("### 🎤 Voice Recording")
        st.info(
            "Click the button below and speak clearly into your microphone. The system will transcribe and analyze your speech.")

        if st.button("🎙️ Start Recording", use_container_width=True):
            with st.spinner("🎤 Recording... Please speak now!"):
                transcript = recognize_speech()

                if transcript.startswith("❌") or transcript.startswith("⚠️"):
                    st.error(transcript)
                else:
                    st.success("✅ Transcription Complete!")
                    st.markdown(f"*Transcribed Text:* {transcript}")

                    with st.spinner("🔄 Analyzing sentiment..."):
                        result = analyze_text_comprehensive(transcript)

                        if result:
                            display_analysis_results(result)
                            save_to_history(result)

    # VIDEO ANALYSIS TAB - COMPLETE FIXED VERSION
    with tab3:
        st.markdown("### 🎥 Video Upload & Analysis")
        st.info("📤 Upload a video file to extract audio, transcribe speech, and analyze sentiment.")

        # System check
        with st.expander("🔍 Check System Requirements"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("*Required:*")
                st.write("✓ FFmpeg installed")
                st.write("✓ Internet connection")
                st.write("✓ Clear audio in video")

            with col2:
                st.markdown("*Supported Formats:*")
                st.write("• MP4, MOV, AVI")
                st.write("• MKV, WEBM")
                st.write("• Max size: 200MB")

        # File uploader
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
            help="Upload a video with clear speech in English"
        )

        if video_file is not None:
            # Display video and info
            col1, col2 = st.columns([3, 2])

            with col1:
                st.video(video_file)

            with col2:
                st.markdown("*📁 File Information*")
                st.write(f"*Name:* {video_file.name}")
                st.write(f"*Size:* {video_file.size / (1024 * 1024):.2f} MB")
                st.write(f"*Type:* {video_file.type}")

                file_extension = video_file.name.split('.')[-1].lower()
                st.write(f"*Format:* {file_extension.upper()}")

            st.markdown("---")

            # Process button
            if st.button("🎬 Process Video & Transcribe", use_container_width=True, type="primary"):

                # Initialize variables
                tmp_video_path = None
                audio_path = None

                try:
                    # STEP 1: Save video to temporary file
                    st.markdown("### 📝 Processing Steps")

                    with st.status("💾 Step 1: Saving video...", expanded=True) as status:
                        # Create temp file with proper extension
                        tmp_video_path = f"temp_video.{file_extension}"

                        # Write uploaded file to disk
                        with open(tmp_video_path, "wb") as f:
                            f.write(video_file.getbuffer())

                        # Verify file was saved
                        if os.path.exists(tmp_video_path):
                            saved_size = os.path.getsize(tmp_video_path)
                            st.write(f"✅ Video saved successfully ({saved_size / (1024 * 1024):.2f} MB)")
                            status.update(label="✅ Step 1: Video saved", state="complete")
                        else:
                            raise Exception("Failed to save video file")

                    # STEP 2: Extract audio
                    with st.status("🎵 Step 2: Extracting audio...", expanded=True) as status:
                        audio_path = extract_audio_from_video(tmp_video_path)

                        if audio_path and os.path.exists(audio_path):
                            status.update(label="✅ Step 2: Audio extracted", state="complete")
                        else:
                            raise Exception("Audio extraction failed")

                    # STEP 3: Transcribe audio
                    with st.status("📝 Step 3: Transcribing audio...", expanded=True) as status:
                        st.write("⏳ This may take 10-30 seconds depending on video length...")

                        transcript = transcribe_audio(audio_path)

                        # Check if transcription was successful
                        if transcript.startswith("❌") or transcript.startswith("⚠️"):
                            status.update(label="⚠️ Step 3: Transcription had issues", state="error")
                            st.error(transcript)

                            st.markdown("---")
                            st.markdown("### 💡 Troubleshooting Tips")
                            st.warning("""
                            *If transcription failed, try these solutions:*

                            1. *Check Video Audio:*
                               - Make sure video has clear speech
                               - Avoid videos with only music or background noise
                               - Ensure audio is in English

                            2. *Test with Simple Video:*
                               - Record a 5-10 second video saying something clearly
                               - Upload and try again

                            3. *Check Internet Connection:*
                               - Google Speech API requires internet
                               - Try again if connection was interrupted
                            """)

                        else:
                            # Successful transcription
                            status.update(label="✅ Step 3: Transcription complete", state="complete")

                            st.success("🎉 Transcription successful!")

                            # Display transcript
                            st.markdown("---")
                            st.markdown("### 📄 Transcribed Text")

                            with st.container():
                                st.text_area(
                                    "Transcript",
                                    value=transcript,
                                    height=150,
                                    disabled=True,
                                    label_visibility="collapsed"
                                )

                                word_count = len(transcript.split())
                                char_count = len(transcript)
                                st.caption(f"📊 {word_count} words • {char_count} characters")

                            # STEP 4: Sentiment Analysis
                            with st.status("🧠 Step 4: Analyzing sentiment...", expanded=True) as status:
                                # Progress bar
                                progress_bar = st.progress(0)
                                for i in range(100):
                                    time.sleep(0.01)
                                    progress_bar.progress(i + 1)

                                # Analyze
                                result = analyze_text_comprehensive(transcript)

                                if result:
                                    status.update(label="✅ Step 4: Analysis complete", state="complete")

                                    st.markdown("---")
                                    display_analysis_results(result)
                                    save_to_history(result)

                                    st.balloons()

                                else:
                                    status.update(label="⚠️ Step 4: Analysis failed", state="error")
                                    st.error("❌ Sentiment analysis failed")

                except Exception as e:
                    st.error(f"❌ *Processing Error:* {str(e)}")
                    st.error(f"*Error Type:* {type(e)._name_}")

                    # Detailed error info
                    with st.expander("🔍 Show detailed error information"):
                        st.code(f"{type(e)._name_}: {str(e)}")

                        st.markdown("*Common Solutions:*")
                        st.write("1. Install FFmpeg: https://ffmpeg.org/download.html")
                        st.write("2. Restart the application after installing FFmpeg")
                        st.write("3. Try a different video file")
                        st.write("4. Ensure video has an audio track")

                finally:
                    # CLEANUP: Remove temporary files
                    try:
                        if tmp_video_path and os.path.exists(tmp_video_path):
                            os.remove(tmp_video_path)
                            st.caption("🗑️ Cleaned up temporary video file")

                        if audio_path and os.path.exists(audio_path):
                            os.remove(audio_path)
                            st.caption("🗑️ Cleaned up temporary audio file")

                    except Exception as cleanup_error:
                        st.caption(f"⚠️ Cleanup note: {cleanup_error}")

        else:
            # Instructions when no file uploaded
            st.markdown("---")
            st.markdown("### 📖 Instructions")
            st.markdown("""
            1. *Upload a video* using the file uploader above
            2. *Click "Process Video"* to start transcription
            3. *Wait for processing* (usually 10-30 seconds)
            4. *View results* including transcript and sentiment analysis

            *Tips for best results:*
            - Use videos with clear, audible speech
            - Avoid videos with loud background music
            - English language works best
            - Keep videos under 5 minutes for faster processing
            """)

elif page == "📚 History":
    st.markdown("## 📚 Analysis History")

    if len(st.session_state.history) == 0:
        st.info("No analyses yet. Start analyzing text, voice, or video content!")
    else:
        st.markdown(f"### Total Analyses: {len(st.session_state.history)}")

        # Convert to DataFrame
        df = pd.DataFrame(st.session_state.history)

        # Visualizations
        st.markdown("### 📊 Visualizations")
        create_visualizations(df)

        # Data Table
        st.markdown("### 📋 Detailed History")

        # Display options
        col1, col2 = st.columns([3, 1])
        with col1:
            search = st.text_input("🔍 Search in history", "")
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()

        # Filter data
        if search:
            filtered_df = df[df['text'].str.contains(search, case=False, na=False)]
        else:
            filtered_df = df

        # Display data
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Analysis #{idx + 1} - {row['final_sentiment']} ({row['timestamp'][:19]})"):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("*Text:*")
                    st.write(row['text'][:200] + "..." if len(row['text']) > 200 else row['text'])

                with col2:
                    st.metric("Sentiment", row['final_sentiment'])
                    st.metric("Score", f"{row['combined_score']:.2f}")
                    st.metric("Confidence", f"{row['confidence']:.2%}")

        # Export
        st.markdown("---")
        if st.button("📥 Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sentiment_analysis_history.csv",
                mime="text/csv"
            )

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About SentiAI")

    st.markdown("""
    ### 🎓 Project Overview

    *SentiAI* is an advanced sentiment analysis platform that combines multiple AI models 
    to provide comprehensive emotional intelligence analysis across text, voice, and video inputs.

    ### 🤖 Technology Stack

    - *BERT*: Deep learning transformer model for contextual sentiment analysis
    - *VADER*: Lexicon-based model optimized for social media text
    - *Google Speech Recognition*: Voice-to-text transcription
    - *Streamlit*: Modern web application framework
    - *Plotly*: Interactive data visualizations
    - *PyDub & FFmpeg*: Audio/video processing

    ### 🎯 Key Features

    1. *Multi-Model Analysis*: Combines BERT and VADER for accuracy
    2. *Multi-Modal Input*: Supports text, voice, and video
    3. *Emotion Detection*: Analyzes 6 core emotions
    4. *Real-Time Processing*: Instant results with progress tracking
    5. *History Management*: Store and review all analyses
    6. *Data Export*: Download results as CSV

    ### 📊 How It Works

    *Text Analysis:*
    - Input is processed through BERT (deep learning) and VADER (lexicon-based)
    - Scores are combined using weighted average (60% BERT, 40% VADER)
    - Final sentiment determined based on combined score thresholds

    *Voice Analysis:*
    - Records audio from microphone
    - Transcribes speech using Google Speech Recognition
    - Analyzes transcribed text for sentiment

    *Video Analysis:*
    - Extracts audio track from video using FFmpeg
    - Converts to optimal format for speech recognition
    - Transcribes and analyzes sentiment

    ### 👥 Team Information

    *Final Year Project*
    - *College:* Your College Name
    - *Team:* Your Team Members
    - *Year:* 2024-2025

    ### 📞 Contact & Support

    For questions or support, please contact your project supervisor or team members.

    ### 📄 License

    This project is developed for educational purposes as part of final year project requirements.

    ---

    *Version:* 1.0.0  
    *Last Updated:* 2024
    """)

    st.markdown("---")
    st.success("✨ Thank you for using SentiAI!")