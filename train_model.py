import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load Data
try:
    df = pd.read_csv('student_depression_dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'student_depression_dataset.csv' not found. Please make sure the file is in the same folder.")
    exit()

# 2. Columns matching Website Form
required_columns = [
    'Gender', 
    'Age', 
    'Academic Pressure', 
    'Work Pressure', 
    'Sleep Duration', 
    'Financial Stress', 
    'Work/Study Hours', 
    'Have you ever had suicidal thoughts ?', 
    'Family History of Mental Illness',
    'Depression' # Target
]

# Filter dataset
try:
    df = df[required_columns].copy()
except KeyError as e:
    print(f"Error: Column not found in dataset: {e}")
    print("Please check your CSV column names.")
    exit()

df.replace('?', np.nan, inplace=True)

# List of columns that MUST be numbers
numeric_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'Financial Stress', 'Work/Study Hours']

# Force these columns to be numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Preprocessing 

# Clean Sleep Duration
df['Sleep Duration'] = df['Sleep Duration'].astype(str).str.replace("'", "").str.strip().str.lower()
sleep_map = {
    "less than 5 hours": 1, "less than 5": 1,
    "5-6 hours": 2, "5-6": 2,
    "7-8 hours": 3, "7-8": 3,
    "more than 8 hours": 4, "more than 8": 4
}
df['Sleep Duration'] = df['Sleep Duration'].map(sleep_map).fillna(2) # Default to 2 if unknown

# Clean Binary Columns (Yes/No -> 1/0)
def clean_binary(x):
    x = str(x).strip().lower()
    if x in ['yes', '1', 'true']: return 1
    return 0

df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].apply(clean_binary)
df['Family History of Mental Illness'] = df['Family History of Mental Illness'].apply(clean_binary)

# Clean Gender (Male->1, Female->0)
df['Gender'] = df['Gender'].apply(lambda x: 1 if str(x).strip().lower().startswith('m') else 0)

# Drop rows where Target (Depression) is missing
df = df.dropna(subset=['Depression'])

# Define Features (X) and Target (y)
X = df.drop('Depression', axis=1)
y = df['Depression']

# 4. Build a Pipeline
numeric_features = ['Age', 'Academic Pressure', 'Work Pressure', 'Financial Stress', 'Work/Study Hours']

# Transformer for numeric columns (Impute missing with Median, then Scale)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
)

# Full Model Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 5. Train
print("Training Logistic Regression Model...")
try:
    model_pipeline.fit(X, y)
    print("Training Complete.")
    
    # 6. Save the model
    joblib.dump(model_pipeline, 'depression_model.pkl')
    print("Model saved as 'depression_model.pkl'")
except Exception as e:
    print(f"An error occurred during training: {e}")