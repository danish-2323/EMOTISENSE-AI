import os
import pandas as pd
from datetime import datetime
from src.config import SESSION_LOGS_DIR, REPORTS_DIR

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(SESSION_LOGS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

def get_timestamp():
    """Get current timestamp"""
    return datetime.now()

def normalize_emotion_scores(emotion_dict):
    """Normalize emotion scores to sum to 1"""
    total = sum(emotion_dict.values())
    if total == 0:
        return {k: 1/len(emotion_dict) for k in emotion_dict}
    return {k: v/total for k, v in emotion_dict.items()}

def calculate_negative_score(emotion_dict):
    """Calculate negative emotion score from emotion dictionary"""
    negative_emotions = ['angry', 'disgust', 'fear', 'sad']
    return sum(emotion_dict.get(emotion, 0) for emotion in negative_emotions)

def save_session_data(df, session_id):
    """Save session data to CSV"""
    ensure_directories()
    filename = f"session_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(SESSION_LOGS_DIR, filename)
    df.to_csv(filepath, index=False)
    return filepath