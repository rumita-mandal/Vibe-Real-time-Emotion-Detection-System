import cv2
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        self.model = DeepFace.build_model("Emotion")

    def analyze_frame(self, frame):
        try:
            prediction = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=False
            )
            if isinstance(prediction, list):
                prediction = prediction[0]

            return prediction["emotion"], prediction["dominant_emotion"]
        except:
            return {}, None

    def preprocess(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
