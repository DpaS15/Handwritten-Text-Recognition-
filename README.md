# 📝 Handwritten Text Recognition Using BiLSTM and CTC
This project is a Flask-based web app that allows users to upload handwritten images or PDFs and converts them into digital text, PDF, and DOCX formats using a deep learning model (CNN + BiLSTM + CTC Loss).

## 🚀 Features
- Upload handwritten `.jpg`, `.jpeg`, `.png`, `.pdf`
- Recognizes handwritten text using a trained deep learning model
- Exports recognized text as:
  - Text (.txt)
  - PDF (.pdf)
  - Word (.docx)
  - Image with text (.jpg)

## 🧠 Tech Stack
- Python, Flask
- TensorFlow / Keras
- BiLSTM + CTC for handwriting recognition
- pdf2image, python-docx, Pillow

## 📦 How to Run
    cd handwritten-text-recognition
    pip install -r requirements.txt
    python app.py
Access the web app at http://127.0.0.1:5000/

## 🧠 Model
The handwriting_model.h5 is trained on the IAM dataset using CNN + BiLSTM + CTC architecture.

## 🙌 Acknowledgements
- IAM Handwriting Dataset
- TensorFlow / Keras
- Flask
