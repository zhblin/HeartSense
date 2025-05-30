import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from collections import deque
from facial_utils import extract_features_and_mask_mediapipe
from model_predictor import HeartRatePredictor
import io

class HeartRateApp:
    def __init__(self, root):
        self.root = root
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        self.root.overrideredirect(True)
        self.root.configure(bg='black')
        self.show_intro()

        self.cap = None
        self.running = False

        self.predictor = HeartRatePredictor("hr_model_cnn_bilstm_attn_10s.h5")
        self.hr_history = deque(maxlen=100)
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.place(x=0, y=0, width=screen_w, height=screen_h)

        self.start_btn = tk.Button(self.root, text="Start", command=self.start_stream, bg='green', fg='white', font=("Arial", 16, "bold"), width=10, height=2)
        self.start_btn.place(x=screen_w - 280, y=screen_h - 70)

        self.quit_btn = tk.Button(self.root, text="Quit", command=self.quit_app, bg='red', fg='white', font=("Arial", 16, "bold"), width=10, height=2)
        self.quit_btn.place(x=screen_w - 140, y=screen_h - 70)

    def show_intro(self):
        intro = tk.Toplevel(self.root)
        intro.title("HeartSense - Introduction")
        intro.geometry("1000x700+100+100")
        intro.configure(bg='white')

        intro.transient(self.root)
        intro.attributes('-topmost', True)
        intro.grab_set()

        title_label = tk.Label(
            intro, text="Welcome to HeartSense",
            font=("Arial", 24, "bold"), bg="white", fg="#333"
        )
        title_label.pack(padx=10, pady=(30, 10))

        paragraph1 = (
            "This desktop application allows you to estimate your heart rate using just your computer's webcam.\n"
            "It uses real-time face tracking and a deep learning model trained to detect subtle color variations\n"
            "on the face caused by blood flow."
        )
        label1 = tk.Label(intro, text=paragraph1, justify="left", font=("Arial", 14), bg='white', fg="#222", wraplength=960)
        label1.pack(padx=20, pady=(0, 20), fill="both")

        how_label = tk.Label(intro, text="✅ How it works:", font=("Arial", 16, "bold"), bg="white", fg="#111")
        how_label.pack(anchor="w", padx=30)

        how_detail = (
            "- Facial landmark detection via MediaPipe\n"
            "- RGB + HSV signal extraction from forehead and cheeks\n"
            "- Deep learning model (CNN + BiLSTM + Attention) for bpm prediction"
        )
        label2 = tk.Label(intro, text=how_detail, justify="left", font=("Arial", 14), bg='white', fg="#222", wraplength=960)
        label2.pack(padx=40, pady=(5, 20), anchor="w")

        contact_label = tk.Label(intro, text="📧 Contact:", font=("Arial", 16, "bold"), bg="white", fg="#111")
        contact_label.pack(anchor="w", padx=30)

        contact_info = (
            "Zhibo Lin | Zhibo.Lin@student.uts.edu.au\n"
            "Yagnika Sindhu Koya | YagnikaSindhu.Koya@student.uts.edu.au"
        )
        label3 = tk.Label(intro, text=contact_info, justify="left", font=("Arial", 14), bg='white', fg="#222", wraplength=960)
        label3.pack(padx=40, pady=(5, 30), anchor="w")

        button = tk.Button(
            intro, text="Continue", command=intro.destroy,
            bg="green", fg="white", font=("Arial", 14, "bold"),
            padx=20, pady=5
        )
        button.pack(pady=10)


    def start_stream(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def quit_app(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.quit()
        self.root.destroy()

    def draw_hr_overlay(self):
        fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
        ax.plot(self.hr_history, color='lime')
        ax.set_facecolor('black')
        fig.patch.set_alpha(0.0)
        ax.set_title("HR Trend", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.set_xlim(0, 100)
        ax.set_ylim(40, 140)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)

    def update_frame(self):
        if not self.running or not self.cap:
            return
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.resize(frame, (self.screen_w, self.screen_h))

        features, mask, boxes = extract_features_and_mask_mediapipe(frame)
        if features is not None:
            self.predictor.update(features, mask)
            for (x, y, w, h) in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            if self.predictor.ready():
                hr = self.predictor.predict()
                self.hr_history.append(hr)
                cv2.putText(frame, f"HR: {hr:.1f} bpm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        if len(self.hr_history) > 5:
            chart = self.draw_hr_overlay().resize((300, 180)).convert("RGBA")
            img.paste(chart, (self.screen_w - 320, 30), chart)

        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, image=imgtk, anchor="nw")
        self.canvas.image = imgtk

        self.root.after(10, self.update_frame)


if __name__ == '__main__':
    root = tk.Tk()
    app = HeartRateApp(root)
    root.mainloop()