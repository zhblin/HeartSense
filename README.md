
# 🧠 Heart Rate Estimation from Facial Video

This project implements a **real-time, contactless heart rate estimation system** using a computer webcam and deep learning. It detects subtle color changes on the human face caused by blood flow and predicts heart rate in bpm.

---

## 🚀 Features

- ✅ Facial landmark detection via **MediaPipe**
- ✅ RGB + HSV signal extraction from **forehead and cheeks**
- ✅ Deep learning model (CNN + BiLSTM + Attention)
- ✅ Real-time heart rate prediction via webcam
- ✅ GUI developed with **Tkinter** + **OpenCV**
- ✅ Exported as a standalone `.exe` application with custom icon

---

## 📂 Project Structure

```
HeartRateApp/
├── gui_app.py                        # Main GUI application
├── facial_utils.py                  # Feature extraction helpers
├── model_predictor.py               # Model inference logic
├── hr_model_cnn_bilstm_attn_10s.h5  # Trained deep learning model
├── shape_predictor_68_face_landmarks.dat  # Dlib facial landmarks
├── HeartSense.ico                   # Custom application icon
├── README.md
```

---

## 🛠 Packaging the Application as `.exe`

You can convert this Python app into a standalone `.exe` file using **PyInstaller**.

### Step-by-step Instructions

#### 1. Install PyInstaller

```bash
pip install pyinstaller
```

#### 2. Run PyInstaller with Required Options

```bash
pyinstaller gui_app.py --onefile --noconsole --icon=HeartSense.ico \
--add-data "hr_model_cnn_bilstm_attn_10s.h5;." \
--add-data "shape_predictor_68_face_landmarks.dat;." \
--add-data "facial_utils.py;." \
--hidden-import=mediapipe.python._framework_bindings
```

#### 🔹 Explanation:

- `--onefile`: create a single `.exe`
- `--noconsole`: hide console window
- `--icon`: **set custom icon (HeartSense.ico)**
- `--add-data`: include model + dependency files
- `--hidden-import`: include MediaPipe bindings

---

### ⚠️ Critical Note: `face_landmark_front_cpu.binarypb`

MediaPipe's Face Mesh requires the binary model file:

```
face_landmark_front_cpu.binarypb
```

After building, you **must manually copy** this file to:

```
dist/gui_app/mediapipe/modules/face_landmark/
```

Find it from your local environment:

```
<venv>/Lib/site-packages/mediapipe/modules/face_landmark/
```

If not copied, the `.exe` will crash with a FileNotFoundError.

---

## 📊 Model Performance

| Metric             | Value         |
|--------------------|---------------|
| MAE                | ~3.87 bpm     |
| RMSE               | ~9.10 bpm     |
| R² Score           | 0.937         |
| ±5 bpm Accuracy    | 73.65% ✅     |

---

## 📧 Contact

> **Zhibo Lin**  
> [Zhibo.Lin@student.uts.edu.au](mailto:Zhibo.Lin@student.uts.edu.au)

> **Yagnika Sindhu Koya**  
> [YagnikaSindhu.Koya@student.uts.edu.au](mailto:YagnikaSindhu.Koya@student.uts.edu.au)

---

## 📜 License

This project is for academic use only. All rights reserved by the authors.
