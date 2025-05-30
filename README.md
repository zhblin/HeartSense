Heart Rate Estimation via Facial Video Signals
This project implements a real-time, contactless heart rate estimation system using webcam video and deep learning. It leverages facial color variations caused by blood flow to predict heart rate (bpm) without any wearable sensors.

The full pipeline includes:\n
Facial landmark detection using MediaPipe
RGB + HSV signal extraction from forehead and cheeks
Signal preprocessing: detrending, filtering, and normalization
Deep learning model (CNN + BiLSTM + Attention)
GUI built with Tkinter + OpenCV for real-time HR display

The system achieves an average error of ~3.87 bpm and ±5 bpm accuracy of 73.65%. It is packaged into an executable desktop application and is suitable for telehealth, fitness tracking, and non-invasive monitoring scenarios.
