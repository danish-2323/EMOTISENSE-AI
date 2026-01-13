import numpy as np
from src.config import FACE_WEIGHT, AUDIO_WEIGHT
from src.utils import calculate_negative_score

class FusionEngine:
    def __init__(self):
        self.face_weight = FACE_WEIGHT
        self.audio_weight = AUDIO_WEIGHT
    
    def fuse_emotions(self, face_emotions, audio_stress_score):
        """Fuse face emotions and audio stress into comprehensive metrics"""
        
        # Calculate face negative score
        face_negative_score = calculate_negative_score(face_emotions)
        
        # Calculate fused metrics with more realistic formulas
        metrics = {}
        
        # Stress: weighted combination with non-linear scaling
        raw_stress = (face_negative_score * self.face_weight + 
                     audio_stress_score * self.audio_weight)
        metrics['stress'] = min(1.0, raw_stress * 1.2)  # Amplify stress signals
        
        # Engagement: based on positive emotions, reduced by stress
        positive_emotions = (face_emotions.get('happy', 0) + 
                           face_emotions.get('surprise', 0) * 0.7)
        base_engagement = positive_emotions * (1.2 - metrics['stress'])
        metrics['engagement'] = max(0.0, min(1.0, base_engagement))
        
        # Confusion: emotion variance + uncertainty indicators
        emotion_values = list(face_emotions.values())
        emotion_variance = np.var(emotion_values)
        uncertainty_factor = 1 - max(emotion_values)  # Low when one emotion dominates
        
        confusion_base = (emotion_variance * 2 + uncertainty_factor * 0.5 + 
                         audio_stress_score * 0.3)
        metrics['confusion'] = max(0.0, min(1.0, confusion_base))
        
        # Confidence: inverse relationship with stress and confusion
        confidence_base = 1.0 - (metrics['stress'] * 0.6 + metrics['confusion'] * 0.4)
        # Add positive emotion boost
        confidence_boost = positive_emotions * 0.3
        metrics['confidence'] = max(0.0, min(1.0, confidence_base + confidence_boost))
        
        # Dominant state with more nuanced rules
        metrics['dominant_state'] = self._determine_dominant_state(
            face_emotions, metrics['stress'], metrics['engagement'], metrics['confusion']
        )
        
        return metrics
    
    def _determine_dominant_state(self, face_emotions, stress, engagement, confusion):
        """Determine dominant emotional state with improved logic"""
        
        # Get dominant face emotion
        dominant_face_emotion = max(face_emotions, key=face_emotions.get)
        dominant_value = face_emotions[dominant_face_emotion]
        
        # Apply hierarchical rules
        if stress > 0.7:
            return 'stressed'
        elif engagement > 0.6 and stress < 0.4:
            return 'engaged'
        elif confusion > 0.6:
            return 'confused'
        elif dominant_value > 0.4:
            if dominant_face_emotion == 'happy':
                return 'positive'
            elif dominant_face_emotion in ['sad', 'angry', 'fear']:
                return 'negative'
            elif dominant_face_emotion == 'neutral' and stress < 0.3:
                return 'calm'
        
        # Default based on overall emotional tone
        positive_score = face_emotions.get('happy', 0) + face_emotions.get('surprise', 0)
        negative_score = calculate_negative_score(face_emotions)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > 0.4:
            return 'negative'
        else:
            return 'neutral'