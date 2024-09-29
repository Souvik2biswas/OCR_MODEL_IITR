# 📄 OCR MULTILINGUAL(using GOT OCR-2.0)
by Souvik Biswas

A simple Gradio app that extracts texts from an uploaded image via GOT OCR-2.0.

Guidelines
Upload your image below and select your preferred mode. Note that more characters may increase wait times.

Plain Texts OCR & Format Texts OCR: Use these modes for basic image-level OCR.
Plain Multi-Crop OCR & Format Multi-Crop OCR: Ideal for images with complex content, offering higher-quality results.
Plain Fine-Grained OCR & Format Fine-Grained OCR: These modes allow you to specify fine-grained regions on the image for more flexible OCR. Regions can be defined by coordinates or colors (red, blue, green, black or white).

[![Open in ](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://document-question-answering-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
