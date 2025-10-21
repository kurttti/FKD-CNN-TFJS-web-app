# FKD CNN TensorFlow.js Web App

This repository contains a Fake Key Detection (FKD) CNN model implemented with TensorFlow.js for web deployment.

## Project Structure

- `build_fkd_app.py` - Main Python application for training and model conversion
- `web/` - Web application directory
  - `index.html` - Main web interface
  - `app.js` - JavaScript application logic
  - `style.css` - Styling for the web interface
  - `model/` - TensorFlow.js model files (excluded from git due to size)

## Setup

1. Install Python dependencies:
   ```bash
   pip install tensorflow tensorflowjs numpy pandas scikit-learn
   ```

2. Run the training script:
   ```bash
   python build_fkd_app.py
   ```

3. Open `web/index.html` in a web browser to use the application.

## Note

Large files (model files and training data) are excluded from this repository due to GitHub's file size limits. These files should be generated locally by running the training script.
