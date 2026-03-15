# AI-Based Handwritten Digit Recognition System
## Setup & Run Guide

---

## QUICK START (3 Steps)

### Step 1 – Install Python packages
Open terminal/command prompt in this folder and run:
```
pip install -r requirements.txt
```

### Step 2 – Train the model
```
python main.py
```
This will:
- Automatically download the MNIST dataset (~12 MB)
- Train the CNN model (~10-15 min on CPU, ~2-3 min on GPU)
- Save the best model as `best_cnn_model.keras`
- Generate all evaluation plots in the `outputs/` folder
- Print test accuracy (~99%)

### Step 3 – Launch the interactive GUI
```
python draw_and_predict.py
```
- Draw any digit (0-9) on the black canvas with your mouse
- The model will predict it in real-time
- You can also load an image file using the "Load Image" button

---

## JUPYTER NOTEBOOK
```
jupyter notebook Digit_Recognition_Notebook.ipynb
```
Run all cells top-to-bottom for a complete step-by-step walkthrough.

---

## FILES EXPLAINED

| File | What it does |
|------|-------------|
| `main.py` | Full pipeline: load data → build CNN → train → evaluate |
| `draw_and_predict.py` | GUI app: draw a digit and get live prediction |
| `Digit_Recognition_Notebook.ipynb` | Jupyter notebook version (for submission) |
| `requirements.txt` | All Python library dependencies |
| `Project_Report.docx` | Complete project report document |

---

## PREDICT YOUR OWN IMAGE

If you have an image of a handwritten digit (PNG/JPG), add this at the bottom of `main.py`:

```python
predict_custom_image(cnn_model, "path/to/your/image.png")
```

**Tips for custom images:**
- Use a dark background with a light-colored digit (like paper)
- The function automatically inverts and resizes the image
- Best results with clean, centered digits

---

## REQUIREMENTS

- Python 3.8+
- ~500 MB disk space (for TensorFlow + MNIST)
- No GPU required (CPU works fine)

---

## EXPECTED RESULTS

| Model | Test Accuracy |
|-------|--------------|
| CNN   | ~99.2%       |
| MLP   | ~97.5%       |
