import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load pre-trained model and scaler (assumes they exist)
# Flowchart: Pre-loaded scaler and Logistic Regression model
model = joblib.load("logistic_regression_diabetes_model.pkl")  # Replace with your model path
scaler = joblib.load("scaler_diabetes.pkl")  # Replace with your scaler path

# Streamlit app title
# Flowchart: User opens the web application
st.title("Diabetes Prediction Web App")

# Input form for health parameters
# Flowchart: User enters health parameters
st.header("Enter Health Parameters")
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Decision threshold slider
# Flowchart: User optionally adjusts decision threshold slider
threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

# Predict button
# Flowchart: User clicks "Predict" button
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, dpf, age]])
    
    # Scale input data
    # Flowchart: System scales input data using pre-loaded scaler
    scaled_data = scaler.transform(input_data)
    
    # Get prediction probability
    # Flowchart: Scaled data passed to Logistic Regression model, model outputs prediction probability
    probability = model.predict_proba(scaled_data)[0][1]  # Probability of positive class (Diabetic)
    
    # Compare with threshold
    # Flowchart: System compares probability with threshold
    prediction = "Diabetic" if probability >= threshold else "Not Diabetic"
    
    # Determine risk level (example thresholds)
    # Flowchart: Display probability score and risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.7:
        risk_level = "Moderate"
    else:
        risk_level = "High"
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Prediction**: {prediction}")
    st.write(f"**Probability of Diabetes**: {probability:.2%}")
    st.write(f"**Risk Level**: {risk_level}")
    
    # Optional feature importance
    # Flowchart: Optionally display feature importance (bar graph)
    if st.checkbox("Show Feature Importance"):
        feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", 
                         "Insulin", "BMI", "Diabetes Pedigree", "Age"]
        coefficients = model.coef_[0]
        
        # Create bar graph
        fig, ax = plt.subplots()
        ax.barh(feature_names, np.abs(coefficients))
        ax.set_xlabel("Feature Importance (Absolute Coefficient)")
        ax.set_title("Feature Importance in Diabetes Prediction")
        st.pyplot(fig)

# Flowchart: End (implicit when app is closed or session ends)