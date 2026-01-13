import sounddevice as sd
import numpy as np
from src.config import AUDIO_SAMPLE_RATE, AUDIO_CHUNK_DURATION, AUDIO_CHANNELS

class MicrophoneCapture:
    def __init__(self):
        self.sample_rate = AUDIO_SAMPLE_RATE
        self.chunk_duration = AUDIO_CHUNK_DURATION
        self.channels = AUDIO_CHANNELS
        self.is_available = self._test_microphone()
    
    def _test_microphone(self):
        """Test if microphone is available"""
        try:
            sd.check_input_settings(channels=self.channels, samplerate=self.sample_rate)
            return True
        except Exception as e:
            print(f"Microphone not available: {e}")
            return False
    
    def capture_audio_chunk(self):
        """Capture audio chunk from microphone"""
        if not self.is_available:
            return self._generate_dummy_audio()
        
        try:
            duration = self.chunk_duration
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            return audio_data.flatten()
        except Exception as e:
            print(f"Audio capture error: {e}")
            return self._generate_dummy_audio()
    
    def _generate_dummy_audio(self):
        """Generate dummy audio data for fallback"""
        duration = self.chunk_duration
        samples = int(duration * self.sample_rate)
        return np.random.normal(0, 0.1, samples).astype(np.float32)