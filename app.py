"""
Airline Passenger Satisfaction Predictor - Streamlit Web App

This interactive web application predicts passenger satisfaction based on
flight and service features using a trained Random Forest model.

Author: ML Course Project Team
Date: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="SkyHigh Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .satisfied {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .dissatisfied {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """
    Load the trained model and preprocessor.
    Uses caching to avoid reloading on every interaction.
    """
    model_path = Path('models/best_model.pkl')
    
    if not model_path.exists():
        st.error("❌ Model file not found!")
        st.info("""
        **Setup Required:**
        1. Run the Jupyter notebook `notebooks/analysis.ipynb`
        2. This will train the model and save it to `models/best_model.pkl`
        3. Then restart this app
        """)
        st.stop()
    
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def create_feature_vector(inputs):
    """
    Create a feature vector from user inputs.
    
    This function maps the user's input to the feature format expected by the model.
    Note: The actual feature engineering should match the preprocessing pipeline.
    
    Args:
        inputs (dict): Dictionary of user inputs
    
    Returns:
        pd.DataFrame: Feature vector ready for prediction
    """
    # Create a dataframe with a single row
    # Feature order must match the training data
    
    # Map Class to ordinal values (Eco=0, Eco Plus=1, Business=2)
    class_mapping = {'Eco': 0, 'Eco Plus': 1, 'Business': 2}
    
    # Map categorical variables
    gender_mapping = {'Male': 0, 'Female': 1}
    customer_type_mapping = {'Disloyal Customer': 0, 'Loyal Customer': 1}
    travel_type_mapping = {'Personal Travel': 0, 'Business travel': 1}
    
    # Create feature dictionary (must match training data exactly!)
    # Feature order and names must match the preprocessing pipeline
    features = {
        'Gender': gender_mapping.get(inputs['gender'], 0),
        'Customer Type': customer_type_mapping.get(inputs['customer_type'], 0),
        'Age': inputs['age'],
        'Type of Travel': travel_type_mapping.get(inputs['travel_type'], 0),
        'Class': class_mapping.get(inputs['class'], 0),
        'Flight Distance': inputs['flight_distance'],
        'Inflight wifi service': inputs['wifi_service'],
        'Departure/Arrival time convenient': inputs.get('time_convenient', 3),  # Added missing feature
        'Ease of Online booking': inputs['online_booking'],
        'Gate location': inputs.get('gate_location', 3),  # Added missing feature
        'Food and drink': inputs['food_drink'],
        'Online boarding': inputs['online_boarding'],
        'Seat comfort': inputs['seat_comfort'],
        'Inflight entertainment': inputs['entertainment'],
        'On-board service': inputs['onboard_service'],
        'Leg room service': inputs['legroom_service'],
        'Baggage handling': inputs['baggage_handling'],
        'Checkin service': inputs['checkin_service'],
        'Inflight service': inputs['inflight_service'],
        'Cleanliness': inputs['cleanliness'],
        'Departure Delay in Minutes': inputs['departure_delay'],
        'Arrival Delay in Minutes': inputs['arrival_delay'],
    }
    
    return pd.DataFrame([features])



    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ SkyHigh Satisfaction Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict passenger satisfaction using machine learning</p>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar for inputs
    st.sidebar.title("📋 Passenger Information")
    st.sidebar.markdown("---")
    
    # Collect user inputs
    inputs = {}
    
    # Demographic Information
    st.sidebar.subheader("👤 Demographics")
    inputs['gender'] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    inputs['age'] = st.sidebar.slider("Age", 7, 85, 40)
    inputs['customer_type'] = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
    
    # Flight Information
    st.sidebar.markdown("---")
    st.sidebar.subheader("✈️ Flight Details")
    inputs['travel_type'] = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    inputs['class'] = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    inputs['flight_distance'] = st.sidebar.number_input("Flight Distance (miles)", 
                                                         min_value=31, max_value=5000, value=1000, step=50)
    
    # Delays
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Delays")
    inputs['departure_delay'] = st.sidebar.number_input("Departure Delay (minutes)", 
                                                          min_value=0, max_value=500, value=0, step=5)
    inputs['arrival_delay'] = st.sidebar.number_input("Arrival Delay (minutes)", 
                                                        min_value=0, max_value=500, value=0, step=5)
    
    # Main content area for service ratings
    st.markdown("## 🌟 Service Ratings")
    st.markdown("*Rate each service from 0 (worst) to 5 (best)*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📡 Connectivity")
        inputs['wifi_service'] = st.slider("Inflight Wifi Service", 0, 5, 3)
        inputs['online_boarding'] = st.slider("Online Boarding", 0, 5, 3)
        inputs['online_booking'] = st.slider("Ease of Online Booking", 0, 5, 3)
        
        st.markdown("### 🍽️ Food & Comfort")
        inputs['food_drink'] = st.slider("Food and Drink", 0, 5, 3)
        inputs['seat_comfort'] = st.slider("Seat Comfort", 0, 5, 3)
        inputs['legroom_service'] = st.slider("Legroom Service", 0, 5, 3)
    
    with col2:
        st.markdown("### 🎬 Entertainment & Service")
        inputs['entertainment'] = st.slider("Inflight Entertainment", 0, 5, 3)
        inputs['onboard_service'] = st.slider("On-board Service", 0, 5, 3)
        inputs['inflight_service'] = st.slider("Inflight Service", 0, 5, 3)
        
        st.markdown("### 🧳 Ground Services")
        inputs['checkin_service'] = st.slider("Check-in Service", 0, 5, 3)
        inputs['baggage_handling'] = st.slider("Baggage Handling", 0, 5, 3)
    
    with col3:
        st.markdown("### 🧹 Cleanliness & Convenience")
        inputs['cleanliness'] = st.slider("Cleanliness", 0, 5, 3)
        inputs['time_convenient'] = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
        inputs['gate_location'] = st.slider("Gate Location Convenience", 0, 5, 3)
        
        st.markdown("### 📊 Quick Ratings")
        st.info(f"""
        **Your Average Rating:** {np.mean([
            inputs['wifi_service'], inputs['online_boarding'], inputs['seat_comfort'],
            inputs['entertainment'], inputs['food_drink'], inputs['legroom_service']
        ]):.1f} / 5.0
        """)
    
    # Prediction button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Predict Satisfaction", use_container_width=True, type="primary")
    
    # Make prediction
    if predict_button:
        try:
            # Load the preprocessor to transform the input data
            preprocessor_path = Path('models/preprocessor.pkl')
            if preprocessor_path.exists():
                preprocessor = joblib.load(preprocessor_path)
                
                # Create a dataframe with the input (must match original column names)
                input_df = pd.DataFrame([{
                    'Gender': inputs['gender'],
                    'Customer Type': inputs['customer_type'],
                    'Age': inputs['age'],
                    'Type of Travel': inputs['travel_type'],
                    'Class': inputs['class'],
                    'Flight Distance': inputs['flight_distance'],
                    'Inflight wifi service': inputs['wifi_service'],
                    'Departure/Arrival time convenient': inputs['time_convenient'],
                    'Ease of Online booking': inputs['online_booking'],
                    'Gate location': inputs['gate_location'],
                    'Food and drink': inputs['food_drink'],
                    'Online boarding': inputs['online_boarding'],
                    'Seat comfort': inputs['seat_comfort'],
                    'Inflight entertainment': inputs['entertainment'],
                    'On-board service': inputs['onboard_service'],
                    'Leg room service': inputs['legroom_service'],
                    'Baggage handling': inputs['baggage_handling'],
                    'Checkin service': inputs['checkin_service'],
                    'Inflight service': inputs['inflight_service'],
                    'Cleanliness': inputs['cleanliness'],
                    'Departure Delay in Minutes': inputs['departure_delay'],
                    'Arrival Delay in Minutes': inputs['arrival_delay'],
                }])
                
                # Transform using the same preprocessing pipeline
                feature_array = preprocessor.preprocessor.transform(input_df)
            else:
                # Fallback if preprocessor not found
                st.error("Preprocessor not found. Please run the training script first.")
                st.stop()
            
            # Make prediction
            prediction = model.predict(feature_array)[0]
            probability = model.predict_proba(feature_array)[0]
            
            # Display result
            st.markdown("---")
            st.markdown("## 🎯 Prediction Result")
            
            if prediction == 1:  # Satisfied
                st.markdown(f"""
                    <div class="prediction-box satisfied">
                        ✅ SATISFIED
                    </div>
                """, unsafe_allow_html=True)
                st.success(f"**Confidence:** {probability[1]*100:.1f}%")
                st.balloons()
            else:  # Dissatisfied
                st.markdown(f"""
                    <div class="prediction-box dissatisfied">
                        ❌ DISSATISFIED
                    </div>
                """, unsafe_allow_html=True)
                st.error(f"**Confidence:** {probability[0]*100:.1f}%")
            
            # Additional insights
            st.markdown("---")
            st.markdown("### 💡 Insights")
            
            col_insight1, col_insight2 = st.columns(2)
            
            with col_insight1:
                st.metric("Satisfaction Probability", f"{probability[1]*100:.1f}%")
                st.metric("Dissatisfaction Probability", f"{probability[0]*100:.1f}%")
            
            with col_insight2:
                # Identify weak points
                service_ratings = {
                    'Online Boarding': inputs['online_boarding'],
                    'Wifi Service': inputs['wifi_service'],
                    'Seat Comfort': inputs['seat_comfort'],
                    'Entertainment': inputs['entertainment'],
                    'Legroom': inputs['legroom_service'],
                }
                
                weak_points = [k for k, v in service_ratings.items() if v < 3]
                strong_points = [k for k, v in service_ratings.items() if v >= 4]
                
                if weak_points:
                    st.warning(f"**Areas for Improvement:** {', '.join(weak_points)}")
                if strong_points:
                    st.success(f"**Strengths:** {', '.join(strong_points)}")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Note: This app requires the model to be trained using the notebook first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
            <p><strong>ML Final Course Project - Airline Passenger Satisfaction</strong></p>
            <p>Powered by Random Forest | Trained on 100K+ passenger reviews</p>
            <p><em>⚠️ Disclaimer: This is an educational project. Predictions are for demonstration purposes only.</em></p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


