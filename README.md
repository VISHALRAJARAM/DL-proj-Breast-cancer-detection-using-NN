🧠 Breast Cancer Detection using Deep Neural Network

A deep learning-based approach for accurately detecting breast cancer from medical features using TensorFlow and Keras.


🚀 Project Overview

This project demonstrates how a Neural Network can be used to classify breast cancer tumors as benign or malignant using the Breast Cancer Wisconsin Diagnostic Dataset. It includes data preprocessing, model building, training, evaluation, and prediction phases in a clean and reproducible manner using Python.


📊 Dataset Used

Source: scikit-learn's built-in load_breast_cancer() dataset
Size: 569 samples, 30 features
Target Classes:
0 → Malignant
1 → Benign


🛠️ What I Did

✅ Data Preprocessing

Loaded the dataset into a Pandas DataFrame
Checked for missing values and understood distribution using descriptive stats
Normalized the features using StandardScaler
✅ Model Building

Built a simple but effective 3-layer Neural Network using TensorFlow + Keras:
Input Layer – Flattened 30 features
Hidden Layer – Dense layer with ReLU activation
Output Layer – Dense layer with Sigmoid activation for binary classification
✅ Model Training

Used sparse_categorical_crossentropy loss and Adam optimizer
Trained the model with validation split to monitor overfitting
Visualized training and validation accuracy and loss
✅ Model Evaluation

Evaluated the model on the test set
Achieved a test accuracy of: ~96.49% 🎯
Converted prediction probabilities into final class labels using np.argmax


🧠 Technologies & Libraries

Python
NumPy
Pandas
Scikit-learn
TensorFlow & Keras
Matplotlib


📈 Results & Insights

Achieved Test Accuracy: 96.49%
Model is able to reliably distinguish between benign and malignant tumors.
Successfully demonstrates the power of deep learning on structured medical data.
The trained model can now be used to predict unseen data entries, aiding diagnostic decisions.


✅ How to Run

Clone the repository
Install required libraries:
pip install numpy pandas scikit-learn tensorflow matplotlib
Run the notebook DL_pro_Breast_cancer_detection_using_NN.ipynb


💼 Why This Project is Impressive

Solid end-to-end deep learning pipeline implementation
Applied best practices in model evaluation and preprocessing
Achieved high performance on a medical dataset, showing real-world impact potential
Clean code, modular design, and insightful visualizations
