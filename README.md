# AI-Based Handwritten Digit Recognition System

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
