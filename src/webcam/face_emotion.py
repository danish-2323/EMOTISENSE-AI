import cv2
import numpy as np
from collections import deque
import os
import warnings

# Suppress all warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Global singleton detector
_fer_detector = None
_opencv_cascade = None

def _get_fer_detector():
    global _fer_detector
    if _fer_detector is None:
        try:
            from fer import FER
            _fer_detector = FER(mtcnn=False)
            print("FER detector initialized")
        except Exception as e:
            print(f"FER initialization failed: {e}")
            _fer_detector = False
    return _fer_detector if _fer_detector is not False else None

def _get_opencv_cascade():
    global _opencv_cascade
    if _opencv_cascade is None:
        try:
            _opencv_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("OpenCV cascade initialized")
        except Exception as e:
            print(f"OpenCV cascade failed: {e}")
            _opencv_cascade = False
    return _opencv_cascade if _opencv_cascade is not False else None

class FaceEmotionDetector:
    def __init__(self):
        self.emotion_history = deque(maxlen=1)  # No smoothing - instant response
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.is_available = True
        print("Face emotion detector ready")
        
    def _detect_face_opencv(self, frame):
        """Detect face using OpenCV"""
        cascade = _get_opencv_cascade()
        if cascade is None:
            return None
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(80, 80))
            
            if len(faces) > 0:
                # Get largest face
                largest = max(faces, key=lambda x: x[2] * x[3])
                return list(largest)
        except Exception as e:
            print(f"Face detection error: {e}")
        return None
    
    def _preprocess_face(self, face_crop):
        """Preprocess face for better emotion recognition"""
        try:
            # Resize to optimal size
            face_resized = cv2.resize(face_crop, (224, 224))
            
            # Convert to grayscale and enhance
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            enhanced = self.clahe.apply(gray)
            
            # Light blur and convert back to BGR
            blurred = cv2.GaussianBlur(enhanced, (3, 3), 0.8)
            processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
            
            return processed
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return cv2.resize(face_crop, (224, 224))
    
    def _get_neutral_output(self, bbox=None):
        """Return neutral emotion output"""
        if bbox is None:
            bbox = [0, 0, 0, 0]
            
        probs = {
            "angry": 0.05,
            "disgust": 0.05,
            "fear": 0.05,
            "happy": 0.15,
            "sad": 0.05,
            "surprise": 0.05,
            "neutral": 0.60
        }
        
        return {
            "emotion": "neutral",
            "confidence": 0.6 if bbox != [0, 0, 0, 0] else 0.0,
            "negative_score": 0.075,
            "bbox": bbox,
            "probs": probs
        }
    
    def detect_emotions(self, frame):
        """Main emotion detection function"""
        if frame is None:
            return self._get_neutral_output()
        
        try:
            # Detect face using OpenCV
            bbox = self._detect_face_opencv(frame)
            
            if bbox is None or bbox[2] < 80 or bbox[3] < 80:
                return self._get_neutral_output()
            
            # Extract and preprocess face
            x, y, w, h = bbox
            face_crop = frame[y:y+h, x:x+w]
            
            if face_crop.size == 0:
                return self._get_neutral_output(bbox)
            
            processed_face = self._preprocess_face(face_crop)
            
            # Get FER predictions
            fer_detector = _get_fer_detector()
            if fer_detector is None:
                return self._get_neutral_output(bbox)
            
            results = fer_detector.detect_emotions(processed_face)
            
            if not results:
                return self._get_neutral_output(bbox)
            
            # Get emotion probabilities
            probs = results[0]['emotions']
            
            # Add to history for smoothing
            self.emotion_history.append(probs)
            
            # Use current frame directly - no smoothing
            avg_probs = probs
            
            # Get dominant emotion and confidence
            max_emotion = max(avg_probs, key=avg_probs.get)
            confidence = avg_probs[max_emotion]
            
            # Very low confidence threshold for maximum detection
            if confidence < 0.15:
                return {
                    "emotion": "neutral",
                    "confidence": confidence,
                    "negative_score": 0.15,
                    "bbox": bbox,
                    "probs": avg_probs
                }
            
            # Calculate negative score
            negative_score = (0.5 * avg_probs.get('sad', 0) + 
                            0.3 * avg_probs.get('angry', 0) + 
                            0.2 * avg_probs.get('fear', 0))
            negative_score = max(0.0, min(1.0, negative_score))
            
            return {
                "emotion": max_emotion,
                "confidence": confidence,
                "negative_score": negative_score,
                "bbox": bbox,
                "probs": avg_probs
            }
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return self._get_neutral_output()
    
    def draw_emotion_box(self, frame, result):
        """Draw bounding box and emotion label on frame"""
        try:
            bbox = result.get('bbox', [0, 0, 0, 0])
            if bbox == [0, 0, 0, 0]:
                return frame
            
            x, y, w, h = bbox
            emotion = result.get('emotion', 'neutral')
            confidence = result.get('confidence', 0.0)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return frame
        except Exception as e:
            print(f"Draw error: {e}")
            return frame