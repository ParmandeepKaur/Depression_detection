# Depression Detection System (Flask + Machine Learning)

## Project Overview
This project is a **Depression Detection System** that uses **Machine Learning** to predict the **risk percentage of depression** based on user inputs.  
The model is integrated with a **Flask web application**, where users can enter data and receive a predicted depression risk score.

This project also demonstrates **DevOps fundamentals** using **GitHub** for version control and project management.

---

## Tech Stack
- **Programming Language:** Python  
- **Backend:** Flask  
- **Machine Learning:** Scikit-learn, NumPy, Pandas  
- **Frontend:** HTML, CSS  
- **DevOps Tool:** GitHub  
- **Model Type:** Predictive Analysis  

---

## Project Structure
```text
Depression_detection/
│
├── model/              # Trained ML model and training scripts
├── templates/          # HTML templates
├── static/             # CSS, JS, images
├── dataset/            # Dataset files (if used)
│
├── app.py              # Flask application entry point
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation
└── .gitignore          # Ignored files for GitHub

Model Training Notebook: `notebooks/Depression_code.ipynb`

## Run with Docker

```bash
docker build -t depression-detection .
docker run -p 5000:5000 depression-detection
