import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent


def load_models():
    model_path = BASE_DIR / "classification_model.pkl"
    scaler_path = BASE_DIR / "classification_scaler.pkl"
    features_path = BASE_DIR / "classification_features.pkl"
    label_encoder_path = BASE_DIR / "classification_label_encoder.pkl"

    required_files = {
        "model": model_path,
        "scaler": scaler_path,
        "features": features_path,
        "label_encoder": label_encoder_path,
    }

    missing = [name for name, path in required_files.items() if not path.exists()]

    if missing:
        st.error(
            "Missing required model files in the app directory: "
            + ", ".join(missing)
            + ". Please save all model artifacts before running the app."
        )
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)
    label_encoder = joblib.load(label_encoder_path)

    return model, scaler, features, label_encoder


st.set_page_config(
    page_title="Supply Chain Risk Predictor",
    page_icon="🚚",
    layout="wide",
)

model, scaler, features, label_encoder = load_models()

st.title("🚚 Supply Chain Risk Prediction System")
st.write(
    "Enter the operational and logistics parameters to predict the delivery risk classification."
)
st.divider()

st.subheader("📊 Input Parameters")
st.info("Enter values using the same units and ranges as the dataset used during training.")

col1, col2 = st.columns(2)

with col1:
    vehicle_gps_latitude = st.number_input("Vehicle GPS Latitude", value=38.0, format="%.6f")
    vehicle_gps_longitude = st.number_input("Vehicle GPS Longitude", value=-90.0, format="%.6f")
    fuel_consumption_rate = st.number_input("Fuel Consumption Rate", min_value=0.0, value=8.0)
    eta_variation_hours = st.number_input("ETA Variation (Hours)", min_value=0.0, value=2.9)
    traffic_congestion_level = st.number_input("Traffic Congestion Level", min_value=0.0, value=5.0)
    warehouse_inventory_level = st.number_input("Warehouse Inventory Level", min_value=0.0, value=300.0)
    loading_unloading_time = st.number_input("Loading / Unloading Time", min_value=0.0, value=2.3)
    handling_equipment_availability = st.number_input(
        "Handling Equipment Availability", min_value=0.0, value=0.30, format="%.4f"
    )
    order_fulfillment_status = st.number_input(
        "Order Fulfillment Status", min_value=0.0, value=0.60, format="%.4f"
    )
    weather_condition_severity = st.number_input(
        "Weather Condition Severity", min_value=0.0, value=0.50, format="%.4f"
    )
    port_congestion_level = st.number_input("Port Congestion Level", min_value=0.0, value=7.0)
    shipping_costs = st.number_input("Shipping Costs", min_value=0.0, value=460.0)

with col2:
    supplier_reliability_score = st.number_input(
        "Supplier Reliability Score", min_value=0.0, value=0.50, format="%.4f"
    )
    lead_time_days = st.number_input("Lead Time (Days)", min_value=0.0, value=5.2)
    historical_demand = st.number_input("Historical Demand", min_value=0.0, value=6000.0)
    iot_temperature = st.number_input("IoT Temperature", value=0.0)
    cargo_condition_status = st.number_input(
        "Cargo Condition Status", min_value=0.0, value=0.30, format="%.4f"
    )
    route_risk_level = st.number_input("Route Risk Level", min_value=0.0, value=7.0)
    customs_clearance_time = st.number_input("Customs Clearance Time", min_value=0.0, value=2.3)
    driver_behavior_score = st.number_input(
        "Driver Behavior Score", min_value=0.0, value=0.50, format="%.4f"
    )
    fatigue_monitoring_score = st.number_input(
        "Fatigue Monitoring Score", min_value=0.0, value=0.60, format="%.4f"
    )
    disruption_likelihood_score = st.number_input(
        "Disruption Likelihood Score", min_value=0.0, value=0.80, format="%.4f"
    )
    delay_probability = st.number_input(
        "Delay Probability", min_value=0.0, max_value=1.0, value=0.70, format="%.4f"
    )

st.divider()
predict_button = st.button("🔮 Predict Delivery Risk", type="primary", use_container_width=True)

if predict_button:
    input_data = pd.DataFrame(
        [
            {
                "vehicle_gps_latitude": vehicle_gps_latitude,
                "vehicle_gps_longitude": vehicle_gps_longitude,
                "fuel_consumption_rate": fuel_consumption_rate,
                "eta_variation_hours": eta_variation_hours,
                "traffic_congestion_level": traffic_congestion_level,
                "warehouse_inventory_level": warehouse_inventory_level,
                "loading_unloading_time": loading_unloading_time,
                "handling_equipment_availability": handling_equipment_availability,
                "order_fulfillment_status": order_fulfillment_status,
                "weather_condition_severity": weather_condition_severity,
                "port_congestion_level": port_congestion_level,
                "shipping_costs": shipping_costs,
                "supplier_reliability_score": supplier_reliability_score,
                "lead_time_days": lead_time_days,
                "historical_demand": historical_demand,
                "iot_temperature": iot_temperature,
                "cargo_condition_status": cargo_condition_status,
                "route_risk_level": route_risk_level,
                "customs_clearance_time": customs_clearance_time,
                "driver_behavior_score": driver_behavior_score,
                "fatigue_monitoring_score": fatigue_monitoring_score,
                "disruption_likelihood_score": disruption_likelihood_score,
                "delay_probability": delay_probability,
            }
        ]
    )

    input_data = input_data[features]
    input_scaled = scaler.transform(input_data)
    prediction_encoded = model.predict(input_scaled)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]

    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction == "High Risk":
        st.error(f"⚠️ Delivery Risk: {prediction}")
    elif prediction == "Moderate Risk":
        st.warning(f"⚠️ Delivery Risk: {prediction}")
    else:
        st.success(f"✅ Delivery Risk: {prediction}")

    with st.expander("Technical Prediction Details"):
        st.write("Encoded Prediction:", prediction_encoded)
        st.write("Decoded Prediction:", prediction)
        st.write("Input Features:", input_data)