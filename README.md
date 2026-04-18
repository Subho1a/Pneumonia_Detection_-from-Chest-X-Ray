# 🫁 Chest X-Ray Pneumonia Detector

A web application built using **Streamlit** and **PyTorch** to detect Pneumonia from Chest X-Ray images using a custom Convolutional Neural Network (CNN).

## 📦 Dataset

The model is trained on a dataset containing NORMAL and PNEUMONIA cases. You can download the exact dataset used for this project from Google Drive here:
[Download Chest X-Ray Dataset](https://drive.google.com/file/d/1-SWJ_nIgotQ11ZHapb-uqWndvzeRs80d/view)

## 🚀 Features

- **Upload X-Rays**: Accept normal image formats (JPG, JPEG, PNG).
- **Deep Learning Model**: Utilizes a PyTorch-based CNN trained specifically on Chest X-Rays.
- **Fast Inference**: Quick processing giving immediate probabilities.
- **Easy Deployment**: Fully ready to run locally or be deployed directly via Streamlit Cloud.

## 🛠️ Project Setup

1. **Clone the repository**
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Set up Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Trained Model**
   Place your heavily trained PyTorch model inside the `checkpoints` directory, specifically named `best_model.pth`.
   ```
   checkpoints/best_model.pth
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 💻 Tech Stack

- **Frontend Environment**: Streamlit
- **Deep Learning Framework**: PyTorch
- **Image Processing**: Torchvision, Pillow (PIL)
- **Language**: Python 3.x

## 📂 Project Structure

```text
Chest_XRay_Pneumonia_Detector/
│
├── app.py                # Main Streamlit web application
├── model.py              # Explicit CNN architecture definition
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .gitignore            # Git ignore configurations
└── checkpoints/
    └── best_model.pth    # Saved best model weights 
```

## 🧠 Model Architecture & Performance

- **Model Performance**: Achieved a peak **Validation Accuracy of ~95.9%** during training phase.
- **Network Depth**: 5 Convolutional Layers equipped with Max Pooling and Batch Normalization.
- **Classification Head**: Fully connected Dense layers condensing features into `NORMAL` and `PNEUMONIA` classes.

## ⚠️ Disclaimer

This application is built for **educational and research purposes only**. It does not substitute professional medical advice, diagnosis, or treatment. Always consult a healthcare professional for medical concerns.
