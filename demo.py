import cv2
from emotion_model import EmotionDetector

detector = EmotionDetector()

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    emotion_probs, dominant = detector.analyze_frame(detector.preprocess(frame))
    print("Dominant Emotion:", dominant)
    print("Probabilities:", emotion_probs)
    print("------------------------------------------------")

    cv2.imshow("Vibe Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
