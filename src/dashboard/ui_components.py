import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def display_metrics_cards(metrics):
    """Display metrics as cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stress Level", f"{metrics['stress']:.2f}", 
                 delta=None, delta_color="inverse")
    
    with col2:
        st.metric("Engagement", f"{metrics['engagement']:.2f}", 
                 delta=None, delta_color="normal")
    
    with col3:
        st.metric("Confusion", f"{metrics['confusion']:.2f}", 
                 delta=None, delta_color="inverse")
    
    with col4:
        st.metric("Confidence", f"{metrics['confidence']:.2f}", 
                 delta=None, delta_color="normal")

def display_dominant_state(dominant_state):
    """Display dominant emotional state"""
    state_colors = {
        'stressed': '🔴',
        'engaged': '🟢', 
        'calm': '🔵',
        'positive': '🟡',
        'negative': '🟠',
        'neutral': '⚪'
    }
    
    icon = state_colors.get(dominant_state, '⚪')
    st.subheader(f"Current State: {icon} {dominant_state.title()}")

def display_stress_alert(stress_history, threshold=0.7, duration=5):
    """Display stress alert if threshold exceeded for duration"""
    if len(stress_history) < duration:
        return False
    
    recent_stress = stress_history[-duration:]
    if all(stress > threshold for stress in recent_stress):
        st.error(f"⚠️ HIGH STRESS ALERT: Stress level above {threshold:.1f} for {duration} seconds!")
        return True
    return False

def display_emotion_breakdown(face_emotions):
    """Display face emotion breakdown"""
    st.subheader("Face Emotion Breakdown")
    
    emotion_df = pd.DataFrame([
        {"Emotion": emotion.title(), "Score": f"{score:.3f}"}
        for emotion, score in face_emotions.items()
    ])
    
    st.dataframe(emotion_df, use_container_width=True)

def display_session_controls():
    """Display session control buttons"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_session = st.button("Start Session", type="primary")
    
    with col2:
        stop_session = st.button("Stop Session", type="secondary")
    
    with col3:
        simulation_mode = st.checkbox("Simulation Mode")
    
    return start_session, stop_session, simulation_mode

def display_about_info():
    """Display about information"""
    st.markdown("""
    ## About EMOTISENSE AI
    
    **EMOTISENSE AI** is a multimodal emotion recognition system that combines:
    
    - 🎥 **Face Emotion Detection**: Real-time facial emotion analysis using computer vision
    - 🎤 **Audio Stress Analysis**: Voice stress detection using audio signal processing
    - 🧠 **Fusion Engine**: Intelligent combination of multiple modalities
    - 📊 **Live Dashboard**: Real-time monitoring and visualization
    - 📈 **Session Reports**: Comprehensive analysis and PDF reports
    
    ### Features:
    - Real-time emotion monitoring
    - Stress level alerts
    - Session logging and reporting
    - Fallback mode for hardware failures
    - Export capabilities (CSV, PDF)
    
    ### Technology Stack:
    - **Frontend**: Streamlit
    - **Computer Vision**: OpenCV, FER
    - **Audio Processing**: librosa, sounddevice
    - **Data Analysis**: pandas, numpy
    - **Visualization**: plotly
    - **Reports**: reportlab
    """)