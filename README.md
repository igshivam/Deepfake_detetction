🔍 DeepFake Detection System

A deep learning-based system designed to detect real vs fake (deepfake) images/videos using the Xception CNN architecture, with preprocessing, evaluation, and deployment via a Streamlit web app.

📌 Project Overview

Deepfakes are AI-generated manipulated media that pose serious threats such as misinformation, identity fraud, and cybercrime. This project builds a robust detection pipeline using deep learning to classify media as Real or Fake.

As described in the project report, the system uses CNN-based feature extraction to detect subtle artifacts like facial inconsistencies and unnatural patterns.

🚀 Features
🎥 Video → Frame → Face Extraction pipeline
🧠 Deep Learning model (Xception) for classification
📊 Model evaluation (Confusion Matrix, ROC Curve, Classification Report)
🌐 Streamlit web app for real-time prediction
📝 Logging system for detection results

🏗️ Project Structure
├── data_preprocessing.py          # Video → frames → faces extraction
├── extractFrames_detectFaces.py   # Alternate preprocessing pipeline
├── train_xception_model.py        # Model training script
├── model_evaluation.py            # Evaluation metrics + plots
├── streamlit_app.py               # Web app for prediction
├── models/
│   └── xception_deepfake_model.h5
├── dataset/
│   ├── videos/
│   └── processed/
├── outputs/
│   ├── confusion_matrix.csv
│   ├── classification_report.csv
│   └── roc_curve.png

⚙️ Tech Stack
Language: Python
Libraries: TensorFlow, Keras, OpenCV, MTCNN, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn
Frontend: Streamlit
Model: Xception (Transfer Learning)
🔄 Workflow
1. Data Preprocessing
Extract frames from videos
Detect faces using MTCNN
Resize to 299×299 for Xception
Save labeled images (Real/Fake)
👉 Implemented in: data_preprocessing

2. Model Training
Pretrained Xception model (ImageNet weights)
Custom dense layers for binary classification
Loss: Binary Crossentropy
Optimizer: Adam
👉 Training pipeline:train_xception_model

3. Model Evaluation
Classification Report
Confusion Matrix
ROC Curve (AUC)
👉 Evaluation script: model_evaluation

4. Deployment
Streamlit app for real-time prediction
Upload image → Get prediction + confidence
👉 App code: streamlit_app

▶️ How to Run
1. Install Dependencies
pip install tensorflow opencv-python mtcnn pillow streamlit sklearn matplotlib seaborn
2. Preprocess Data
python data_preprocessing.py
3. Train Model
python train_xception_model.py
4. Evaluate Model
python model_evaluation.py
5. Run Web App
streamlit run streamlit_app.py

📌 Conclusion
This project demonstrates a complete deepfake detection pipeline from preprocessing to deployment.

## 👨‍💻 Author

**Shivam Narayan


---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
