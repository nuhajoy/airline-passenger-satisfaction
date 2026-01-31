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
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="SkyHigh Satisfaction Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Simple & Clean UI Styling (UI ONLY)
# --------------------------------------------------
st.markdown("""
<style>
/* Main title */
.main-header {
    font-size: 2.6rem;
    color: #1f3c88;
    text-align: center;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

/* Subtitle */
.sub-header {
    text-align: center;
    color: #5a6f8f;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* Prediction result box */
.prediction-box {
    padding: 1.8rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 600;
    margin: 2rem auto;
    max-width: 500px;
}

/* Satisfied style */
.satisfied {
    background-color: #e8f5e9;
    color: #1b5e20;
    border-left: 6px solid #2e7d32;
}

/* Dissatisfied style */
.dissatisfied {
    background-color: #fdecea;
    color: #7f1d1d;
    border-left: 6px solid #c62828;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #f7f9fc;
}

/* Buttons */
button[kind="primary"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
}

/* Input spacing */
.stSlider, .stNumberInput, .stSelectbox {
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load model (cached)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/best_model.pkl")

    if not model_path.exists():
        st.error("❌ Model file not found!")
        st.info("""
        **Setup Required:**
        1. Run the training notebook/script
        2. Save model to `models/best_model.pkl`
        3. Restart the Streamlit app
        """)
        st.stop()

    return joblib.load(model_path)


# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():

    # Header
    st.markdown('<h1 class="main-header">✈️ SkyHigh Satisfaction Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict airline passenger satisfaction using machine learning</p>', unsafe_allow_html=True)

    # Load model
    model = load_model()

    # Sidebar inputs
    st.sidebar.title("📋 Passenger Information")
    st.sidebar.markdown("---")

    inputs = {}

    # Demographics
    st.sidebar.subheader("👤 Demographics")
    inputs["gender"] = st.sidebar.selectbox("Gender", ["Male", "Female"])
    inputs["age"] = st.sidebar.slider("Age", 7, 85, 40)
    inputs["customer_type"] = st.sidebar.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])

    # Flight info
    st.sidebar.markdown("---")
    st.sidebar.subheader("✈️ Flight Details")
    inputs["travel_type"] = st.sidebar.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
    inputs["class"] = st.sidebar.selectbox("Class", ["Eco", "Eco Plus", "Business"])
    inputs["flight_distance"] = st.sidebar.number_input(
        "Flight Distance (miles)", min_value=31, max_value=5000, value=1000, step=50
    )

    # Delays
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏱️ Delays")
    inputs["departure_delay"] = st.sidebar.number_input(
        "Departure Delay (minutes)", min_value=0, max_value=500, value=0, step=5
    )
    inputs["arrival_delay"] = st.sidebar.number_input(
        "Arrival Delay (minutes)", min_value=0, max_value=500, value=0, step=5
    )

    # Main ratings
    st.markdown("## 🌟 Service Ratings")
    st.markdown("*Rate each service from 0 (worst) to 5 (best)*")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📡 Connectivity")
        inputs["wifi_service"] = st.slider("Inflight Wifi Service", 0, 5, 3)
        inputs["online_boarding"] = st.slider("Online Boarding", 0, 5, 3)
        inputs["online_booking"] = st.slider("Ease of Online Booking", 0, 5, 3)

        st.markdown("### 🍽️ Comfort")
        inputs["food_drink"] = st.slider("Food and Drink", 0, 5, 3)
        inputs["seat_comfort"] = st.slider("Seat Comfort", 0, 5, 3)
        inputs["legroom_service"] = st.slider("Legroom Service", 0, 5, 3)

    with col2:
        st.markdown("### 🎬 Entertainment & Service")
        inputs["entertainment"] = st.slider("Inflight Entertainment", 0, 5, 3)
        inputs["onboard_service"] = st.slider("On-board Service", 0, 5, 3)
        inputs["inflight_service"] = st.slider("Inflight Service", 0, 5, 3)

        st.markdown("### 🧳 Ground Services")
        inputs["checkin_service"] = st.slider("Check-in Service", 0, 5, 3)
        inputs["baggage_handling"] = st.slider("Baggage Handling", 0, 5, 3)

    with col3:
        st.markdown("### 🧹 Cleanliness & Convenience")
        inputs["cleanliness"] = st.slider("Cleanliness", 0, 5, 3)
        inputs["time_convenient"] = st.slider("Departure/Arrival Time Convenient", 0, 5, 3)
        inputs["gate_location"] = st.slider("Gate Location Convenience", 0, 5, 3)

        st.info(f"""
        **Average Rating:** {np.mean([
            inputs["wifi_service"], inputs["online_boarding"],
            inputs["seat_comfort"], inputs["entertainment"],
            inputs["food_drink"], inputs["legroom_service"]
        ]):.1f} / 5
        """)

    # Predict button
    st.markdown("---")
    _, col_btn, _ = st.columns([1, 1, 1])
    with col_btn:
        predict_button = st.button("🔮 Predict Satisfaction", use_container_width=True, type="primary")

    # Prediction
    if predict_button:
        preprocessor_path = Path("models/preprocessor.pkl")

        if not preprocessor_path.exists():
            st.error("Preprocessor not found. Please run preprocessing first.")
            st.stop()

        preprocessor = joblib.load(preprocessor_path)

        input_df = pd.DataFrame([{
            "Gender": inputs["gender"],
            "Customer Type": inputs["customer_type"],
            "Age": inputs["age"],
            "Type of Travel": inputs["travel_type"],
            "Class": inputs["class"],
            "Flight Distance": inputs["flight_distance"],
            "Inflight wifi service": inputs["wifi_service"],
            "Departure/Arrival time convenient": inputs["time_convenient"],
            "Ease of Online booking": inputs["online_booking"],
            "Gate location": inputs["gate_location"],
            "Food and drink": inputs["food_drink"],
            "Online boarding": inputs["online_boarding"],
            "Seat comfort": inputs["seat_comfort"],
            "Inflight entertainment": inputs["entertainment"],
            "On-board service": inputs["onboard_service"],
            "Leg room service": inputs["legroom_service"],
            "Baggage handling": inputs["baggage_handling"],
            "Checkin service": inputs["checkin_service"],
            "Inflight service": inputs["inflight_service"],
            "Cleanliness": inputs["cleanliness"],
            "Departure Delay in Minutes": inputs["departure_delay"],
            "Arrival Delay in Minutes": inputs["arrival_delay"],
        }])

        X = preprocessor.preprocessor.transform(input_df)
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        st.markdown("## 🎯 Prediction Result")

        if prediction == 1:
            st.markdown(
                '<div class="prediction-box satisfied">✅ SATISFIED</div>',
                unsafe_allow_html=True
            )
            st.success(f"Confidence: {probability[1]*100:.1f}%")
            st.balloons()
        else:
            st.markdown(
                '<div class="prediction-box dissatisfied">❌ DISSATISFIED</div>',
                unsafe_allow_html=True
            )
            st.error(f"Confidence: {probability[0]*100:.1f}%")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#7f8c8d; padding:1.5rem;">
        <strong>ML Final Course Project – Airline Passenger Satisfaction</strong><br>
        Powered by Random Forest | Educational Use Only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()