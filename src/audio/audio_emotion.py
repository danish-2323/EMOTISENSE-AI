import numpy as np
import librosa
from src.config import AUDIO_SAMPLE_RATE

class AudioEmotionAnalyzer:
    def __init__(self):
        self.sample_rate = AUDIO_SAMPLE_RATE
    
    def analyze_stress(self, audio_data):
        """Analyze stress level from audio data"""
        try:
            # Extract audio features
            features = self._extract_features(audio_data)
            
            # Calculate stress score using heuristic
            stress_score = self._calculate_stress_score(features)
            
            return max(0.0, min(1.0, stress_score))
        except Exception as e:
            print(f"Audio analysis error: {e}")
            return 0.5  # Default stress score
    
    def _extract_features(self, audio_data):
        """Extract audio features using librosa"""
        features = {}
        
        # RMS Energy
        rms = librosa.feature.rms(y=audio_data)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zcr)
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        
        return features
    
    def _calculate_stress_score(self, features):
        """Calculate stress score from features using heuristic approach"""
        # Simple heuristic based on audio characteristics
        # Higher RMS, ZCR, and spectral centroid often indicate stress/excitement
        
        rms_score = min(features['rms_mean'] * 10, 1.0)  # Normalize RMS
        zcr_score = min(features['zcr_mean'] * 5, 1.0)   # Normalize ZCR
        
        # MFCC-based score (higher variance in lower coefficients indicates stress)
        mfcc_score = min(np.mean(features['mfcc_std'][:5]) / 10, 1.0)
        
        # Spectral centroid score
        spectral_score = min(features['spectral_centroid_mean'] / 4000, 1.0)
        
        # Weighted combination
        stress_score = (rms_score * 0.3 + zcr_score * 0.2 + 
                       mfcc_score * 0.3 + spectral_score * 0.2)
        
        return stress_score