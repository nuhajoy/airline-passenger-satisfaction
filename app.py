import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Airline Passenger Satisfaction Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airline Passenger Satisfaction Prediction")
st.write(
    "This app predicts whether a passenger is **Satisfied** or "
    "**Neutral / Dissatisfied** based on flight and service details."
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/satisfaction_model.pkl")

model = load_model()

# --------------------------------------------------
# User Input Section
# --------------------------------------------------
st.header("Passenger Information")

gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)

type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
travel_class = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])

flight_distance = st.number_input("Flight Distance (km)", min_value=50, max_value=5000, value=800)
departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, max_value=1000, value=0)
arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, max_value=1000, value=0)

st.header("Service Ratings (0 = Worst, 5 = Best)")

wifi = st.slider("In-flight Wifi Service", 0, 5, 3)
seat_comfort = st.slider("Seat Comfort", 0, 5, 3)
online_boarding = st.slider("Online Boarding", 0, 5, 3)
cleanliness = st.slider("Cleanliness", 0, 5, 3)
leg_room = st.slider("Leg Room Service", 0, 5, 3)
food_drink = st.slider("Food and Drink", 0, 5, 3)
inflight_service = st.slider("In-flight Service", 0, 5, 3)
baggage_handling = st.slider("Baggage Handling", 0, 5, 3)
checkin_service = st.slider("Check-in Service", 0, 5, 3)

# --------------------------------------------------
# Prepare Input Data
# --------------------------------------------------
input_data = pd.DataFrame([{
    "Gender": gender,
    "Customer Type": customer_type,
    "Age": age,
    "Type of Travel": type_of_travel,
    "Class": travel_class,
    "Flight Distance": flight_distance,
    "Departure Delay in Minutes": departure_delay,
    "Arrival Delay in Minutes": arrival_delay,
    "Inflight wifi service": wifi,
    "Seat comfort": seat_comfort,
    "Online boarding": online_boarding,
    "Cleanliness": cleanliness,
    "Leg room service": leg_room,
    "Food and drink": food_drink,
    "Inflight service": inflight_service,
    "Baggage handling": baggage_handling,
    "Checkin service": checkin_service
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Satisfaction"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(f"✅ Passenger is **Satisfied** (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ Passenger is **Neutral / Dissatisfied** (Confidence: {1 - probability:.2%})")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Machine Learning Lab Final Project | Airline Passenger Satisfaction")
