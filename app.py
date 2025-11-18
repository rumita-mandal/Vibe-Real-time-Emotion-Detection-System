import streamlit as st
import cv2
import time
from PIL import Image
import matplotlib.pyplot as plt
import io
from emotion_model import EmotionDetector

st.set_page_config(page_title="Vibe – Emotion Detection", layout="wide")

st.title("🎭 Vibe – Real-Time Emotion Detection System")

detector = EmotionDetector()

col1, col2 = st.columns([2, 1])
camera_display = col1.empty()
chart_display = col2.empty()
caption_display = col2.empty()

run = st.button("Start Webcam")
stop = st.button("Stop Webcam")

running = False

if run:
    running = True
if stop:
    running = False

def emotion_chart(emotion_dict):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(list(emotion_dict.keys()), list(emotion_dict.values()))
    ax.set_xlabel("Probability (%)")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

if running:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = detector.preprocess(frame)
        probs, dominant = detector.analyze_frame(rgb)

        camera_display.image(rgb, channels="RGB")

        if dominant:
            caption_display.markdown(f"### Dominant Emotion: **{dominant.upper()}**")

        if probs:
            chart = emotion_chart(probs)
            chart_display.image(chart)

        if stop:
            running = False
            break

        time.sleep(0.05)

    cap.release()
