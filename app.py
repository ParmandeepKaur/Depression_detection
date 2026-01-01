from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = 'depression_model.pkl'

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    model = None
    print("Warning: 'depression_model.pkl' not found. Please run train_model.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/screen', methods=['POST'])
def screen():
    if not model:
        return render_template('result.html', risk_level="Error", score=0, error="Model not loaded")

    try:
        # 1. Extract and Map Data
        data = request.form
        
        # Gender: Male=1, Female=0
        gender = 1 if data.get('gender') == 'Male' else 0
        
        # Sleep: Map to 1-4 scale
        sleep_str = data.get('sleep_duration', '')
        if 'Less' in sleep_str: sleep_val = 1
        elif '5' in sleep_str: sleep_val = 2
        elif '7' in sleep_str: sleep_val = 3
        else: sleep_val = 4 
        
        # Pressures: Low=1, Medium=3, High=5 
        def map_pressure(level):
            if level == 'Low': return 1
            if level == 'Medium': return 3
            return 5
            
        ac_pressure = map_pressure(data.get('academic_pressure'))
        wk_pressure = map_pressure(data.get('work_pressure'))
        fin_stress = map_pressure(data.get('financial_stress'))
        
        # Work Hours
        wh_str = data.get('work_hours', '')
        if 'Less' in wh_str: work_hours = 2
        elif '4-8' in wh_str: work_hours = 6
        else: work_hours = 10
        
        # Binary Questions
        suicidal = 1 if data.get('suicidal_thoughts') == 'Yes' else 0
        history = 1 if data.get('family_history') == 'Yes' else 0
        
        age = float(data.get('age'))

        # 2. DataFrame for Prediction
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Academic Pressure': ac_pressure,
            'Work Pressure': wk_pressure,
            'Sleep Duration': sleep_val,
            'Financial Stress': fin_stress,
            'Work/Study Hours': work_hours,
            'Have you ever had suicidal thoughts ?': suicidal,
            'Family History of Mental Illness': history
        }])

        # 3. Predict
        probability = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Depression)
        risk_score = int(probability * 100) # Convert 0.85 -> 85
        
        # Model working
        print(f"\nMODEL PREDICTION:")
        print(f"Input Data: {input_data.to_dict(orient='records')}")
        print(f"Raw Probability: {probability:.4f}")
        print(f"Calculated Score: {risk_score}%")
        print("-" * 30)

        # Determine Risk Level
        if risk_score < 30:
            risk_level = "Low Risk"
        elif risk_score < 70:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        return render_template('result.html', risk_level=risk_level, score=risk_score)

    except Exception as e:
        print(f"Error: {e}")
        return f"An error occurred: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)