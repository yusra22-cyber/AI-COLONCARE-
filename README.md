# AI-Based Colorectal Polyp Detection – ColonCare
This is my final year project developed as part of my Software Engineering degree. ColonCare is an AI-based healthcare system designed to detect and classify colorectal polyps (such as tubular, villous, serrated, etc.) using deep learning techniques.

## 🧠 Project Summary
- Utilized a Swin Transformer model for image classification.
- Implemented the project in Python with PyTorch for model training.
- Extracted video frames from polyp videos and categorized them manually into 5 types.
- Trained and tested the model using a custom-labeled dataset.

## 🖥️ Web Application
Developed a full-stack web app using:
  - **Flask** (backend)
  - **Tailwind CSS + HTML** (frontend)
  - **SQLite** for user login & registration
- Integrated the trained model into the Flask app for real-time prediction.
- Users can upload polyp images and get results directly in the browser.
