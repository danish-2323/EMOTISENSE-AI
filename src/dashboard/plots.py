import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

def create_timeline_chart(df, timeline_seconds=60):
    """Create timeline chart for last N seconds"""
    if df.empty:
        return go.Figure()
    
    # Filter last N seconds
    current_time = datetime.now()
    cutoff_time = current_time - timedelta(seconds=timeline_seconds)
    recent_df = df[df['timestamp'] >= cutoff_time].copy()
    
    if recent_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Add stress line
    fig.add_trace(go.Scatter(
        x=recent_df['timestamp'],
        y=recent_df['stress'],
        mode='lines+markers',
        name='Stress',
        line=dict(color='red', width=2)
    ))
    
    # Add engagement line
    fig.add_trace(go.Scatter(
        x=recent_df['timestamp'],
        y=recent_df['engagement'],
        mode='lines+markers',
        name='Engagement',
        line=dict(color='green', width=2)
    ))
    
    # Add confidence line
    fig.add_trace(go.Scatter(
        x=recent_df['timestamp'],
        y=recent_df['confidence'],
        mode='lines+markers',
        name='Confidence',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=f"Emotion Timeline (Last {timeline_seconds}s)",
        xaxis_title="Time",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig

def create_emotion_pie_chart(face_emotions):
    """Create pie chart for face emotions"""
    emotions = list(face_emotions.keys())
    values = list(face_emotions.values())
    
    fig = px.pie(
        values=values,
        names=emotions,
        title="Current Face Emotions"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig

def create_stress_gauge(stress_level):
    """Create gauge chart for stress level"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = stress_level,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stress Level"},
        delta = {'reference': 0.5},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.3], 'color': "lightgreen"},
                {'range': [0.3, 0.7], 'color': "yellow"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_session_summary_chart(df):
    """Create session summary chart"""
    if df.empty:
        return go.Figure()
    
    # Calculate summary statistics
    summary_stats = {
        'Average Stress': df['stress'].mean(),
        'Average Engagement': df['engagement'].mean(),
        'Average Confidence': df['confidence'].mean(),
        'Average Confusion': df['confusion'].mean()
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(summary_stats.keys()),
            y=list(summary_stats.values()),
            marker_color=['red', 'green', 'blue', 'orange']
        )
    ])
    
    fig.update_layout(
        title="Session Summary",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    return fig