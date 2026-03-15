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


This project implements a complete deep learning pipeline for recognizing handwritten digits using a Convolutional Neural Network (CNN). The model is trained on the MNIST dataset and achieves ~99% accuracy on unseen test data.

The system demonstrates the full machine learning workflow including data preprocessing, CNN model training, evaluation, and real-time inference through an interactive graphical user interface.

Users can draw digits directly on a canvas and the trained model predicts the digit along with a confidence score and probability distribution.

---

## Features

- Convolutional Neural Network (CNN) for image classification
- Training on the MNIST dataset (70,000 handwritten digits)
- ~99% test accuracy
- Training visualization (accuracy & loss curves)
- Confusion matrix and evaluation metrics
- Sample prediction visualization
- Interactive Tkinter GUI for real-time digit recognition
- Model comparison with a Multilayer Perceptron (MLP)

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow (PIL)
- Tkinter

---

## Project Pipeline

The project follows a standard deep learning workflow:

1. MNIST dataset is loaded using TensorFlow.
2. Images are normalized and reshaped for CNN input.
3. A convolutional neural network extracts spatial features.
4. The model is trained using backpropagation and gradient descent.
5. Evaluation metrics and visualizations are generated.
6. The trained model is deployed through a GUI for real-time predictions.

---

## Model Architecture

The CNN architecture consists of:

- Convolutional layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Dropout layers for regularization
- Fully connected dense layers for classification
- Softmax output layer for probability prediction

The model learns hierarchical image features such as edges, curves, and digit shapes.

---

## Running the Project

Install dependencies:

pip install -r requirements.txt


Train the model:
python main.py


Run the interactive digit recognition interface:
python draw_and_predict.py


---

## Output Visualizations

Training generates:

- Accuracy and loss curves
- Confusion matrix
- Sample prediction grid
- Misclassified examples

These outputs are stored in the `outputs/` folder.

---

## Results

- Test Accuracy: ~99%
- Training dataset: 60,000 images
- Test dataset: 10,000 images
- Image size: 28×28 grayscale

---

## Applications

Handwritten digit recognition is used in many real-world systems:

- Postal code recognition
- Bank cheque processing
- Document digitization
- Form automation systems

---

## Future Improvements

- Data augmentation for better handwriting generalization
- Web-based interface using Flask or FastAPI
- Multi-digit recognition
- Training on EMNIST dataset for full character recognition

---

## Author

Zaheer Abbas  
B.Tech CSE (AI/ML)

