"""
Draw-and-Predict GUI
====================
Draw a digit with your mouse → CNN predicts it live.
Requirements: tkinter (built into Python), tensorflow, numpy, pillow
Run:  python draw_and_predict.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
import os

# Try loading a saved model; fall back with instructions if not found
try:
    import tensorflow as tf
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False

CANVAS_SIZE  = 280   # 280×280 drawing canvas (10× the 28×28 input)
BRUSH_RADIUS = 14    # brush width


class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.resizable(False, False)

        self.model = None
        self._load_model()

        # PIL image for the drawing canvas
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)

        self._build_ui()

    # ── Model ──────────────────────────────────────────────
    def _load_model(self):
        if not MODEL_AVAILABLE:
            return
        for name in ["best_cnn_model.keras", "best_cnn_model.h5"]:
            if os.path.exists(name):
                self.model = tf.keras.models.load_model(name)
                print(f"Model loaded: {name}")
                return
        print("⚠️  No saved model found. Train the model first (main.py).")

    # ── UI ─────────────────────────────────────────────────
    def _build_ui(self):
        # ── Title bar ──
        title = tk.Label(self.root, text="✏️  Handwritten Digit Recognizer",
                         font=("Arial", 16, "bold"), pady=8)
        title.pack()

        main_frame = tk.Frame(self.root, padx=12, pady=6)
        main_frame.pack()

        # ── Drawing canvas ──
        left = tk.Frame(main_frame)
        left.pack(side=tk.LEFT, padx=6)

        tk.Label(left, text="Draw a digit (0–9):", font=("Arial", 11)).pack(anchor='w')

        self.canvas = tk.Canvas(left, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", cursor="crosshair",
                                highlightbackground="grey", highlightthickness=2)
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        btn_frame = tk.Frame(left)
        btn_frame.pack(pady=6)

        tk.Button(btn_frame, text="🗑  Clear", width=10,
                  command=self.clear, bg="#e74c3c", fg="white",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="🔍 Predict", width=10,
                  command=self.predict, bg="#2ecc71", fg="white",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_frame, text="📂 Load Image", width=12,
                  command=self.load_image, font=("Arial", 10)).pack(side=tk.LEFT, padx=4)

        # ── Result panel ──
        right = tk.Frame(main_frame, padx=12)
        right.pack(side=tk.LEFT, anchor='n')

        tk.Label(right, text="Prediction", font=("Arial", 13, "bold")).pack(pady=(0, 6))

        self.result_label = tk.Label(right, text="—", font=("Arial", 64, "bold"),
                                     width=3, height=1, relief="groove",
                                     bg="white", fg="#2c3e50")
        self.result_label.pack()

        self.conf_label = tk.Label(right, text="Confidence: —",
                                   font=("Arial", 11), fg="#555")
        self.conf_label.pack(pady=4)

        # Bar chart for all probabilities
        tk.Label(right, text="Class Probabilities:", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        self.bars_frame = tk.Frame(right, bg="white", relief="sunken", bd=1,
                                   width=220, height=200)
        self.bars_frame.pack()
        self.bars_frame.pack_propagate(False)
        self._init_bars()

        # ── Preview of preprocessed image ──
        tk.Label(right, text="28×28 Input:", font=("Arial", 10, "bold")).pack(pady=(10, 2))
        self.preview_label = tk.Label(right, bg="#ccc", width=56, height=56)
        self.preview_label.pack()

        # ── Status ──
        self.status = tk.StringVar(value="Ready – draw a digit then press Predict")
        tk.Label(self.root, textvariable=self.status, font=("Arial", 9),
                 fg="#666", pady=4).pack()

    def _init_bars(self):
        """Create 10 probability bar labels."""
        self.bar_vars  = []
        self.bar_fills = []
        for d in range(10):
            row = tk.Frame(self.bars_frame, bg="white")
            row.pack(fill='x', padx=4, pady=1)
            tk.Label(row, text=str(d), width=2, font=("Arial", 9), bg="white").pack(side=tk.LEFT)
            bar_bg = tk.Frame(row, bg="#ddd", height=14, width=150)
            bar_bg.pack(side=tk.LEFT, padx=2)
            bar_bg.pack_propagate(False)
            fill = tk.Frame(bar_bg, bg="#3498db", height=14, width=0)
            fill.place(x=0, y=0, height=14)
            self.bar_fills.append(fill)
            pct_lbl = tk.Label(row, text="0%", width=5, font=("Arial", 9), bg="white")
            pct_lbl.pack(side=tk.LEFT)
            self.bar_vars.append(pct_lbl)

    def _update_bars(self, probs):
        for d in range(10):
            pct   = probs[d] * 100
            width = int(pcts := probs[d] * 150)
            color = "#2ecc71" if d == np.argmax(probs) else "#3498db"
            self.bar_fills[d].place(x=0, y=0, width=width, height=14)
            self.bar_fills[d].configure(bg=color)
            self.bar_vars[d].configure(text=f"{pct:.1f}%")

    # ── Drawing ────────────────────────────────────────────
    def _paint(self, event):
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def _on_release(self, _event):
        if self.model:
            self.predict()

    def clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        self.result_label.configure(text="—")
        self.conf_label.configure(text="Confidence: —")
        for d in range(10):
            self.bar_fills[d].place(x=0, y=0, width=0, height=14)
            self.bar_vars[d].configure(text="0%")
        self.status.set("Canvas cleared")

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if not path:
            return
        img = Image.open(path).convert("L").resize((CANVAS_SIZE, CANVAS_SIZE))
        self.pil_image = img
        self.pil_draw  = ImageDraw.Draw(self.pil_image)
        # Show on canvas
        photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor="nw", image=photo)
        self.predict()

    # ── Prediction ─────────────────────────────────────────
    def _preprocess(self):
        img = self.pil_image.resize((28, 28), Image.LANCZOS)
        arr = np.array(img).astype("float32") / 255.0
        return arr.reshape(1, 28, 28, 1), img

    def predict(self):
        if not MODEL_AVAILABLE:
            messagebox.showerror("TensorFlow Missing",
                                 "Install TensorFlow:\n  pip install tensorflow")
            return
        if self.model is None:
            messagebox.showerror("No Model",
                                 "Train the model first:\n  python main.py\n\n"
                                 "Then re-run this app.")
            return

        arr, thumb = self._preprocess()

        # Show 28×28 preview
        preview_big = thumb.resize((56, 56), Image.NEAREST)
        tk_img = ImageTk.PhotoImage(preview_big)
        self.preview_label.configure(image=tk_img, bg="black")
        self.preview_label.image = tk_img

        probs     = self.model.predict(arr, verbose=0)[0]
        digit     = int(np.argmax(probs))
        conf      = probs[digit] * 100

        self.result_label.configure(text=str(digit))
        self.conf_label.configure(text=f"Confidence: {conf:.1f}%")
        self._update_bars(probs)
        self.status.set(f"Predicted: {digit}  ({conf:.1f}% confidence)")


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = DigitRecognizerApp(root)
    root.mainloop()
