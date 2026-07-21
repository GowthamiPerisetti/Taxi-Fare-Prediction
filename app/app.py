import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Taxi Fare Prediction System",
    page_icon="🚖",
    layout="wide"
)

# ==========================================================
# Load Model
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "taxi_fare_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_importance.csv"

model = joblib.load(MODEL_PATH)
feature_importance = pd.read_csv(FEATURE_PATH)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("🚖 Taxi Fare Prediction")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.success("Random Forest Regressor")

st.sidebar.markdown("### Performance")

st.sidebar.metric("R² Score", "95.39%")
st.sidebar.metric("MAE", "0.3454")
st.sidebar.metric("RMSE", "2.5174")

st.sidebar.markdown("---")

st.sidebar.info(
    """
Predict taxi fares using a Machine Learning model trained on
200,000 NYC taxi trips.
"""
)

# ==========================================================
# Title
# ==========================================================

st.title("🚖 Taxi Fare Prediction System")

st.write(
    "Predict the taxi fare using a trained Random Forest Machine Learning model."
)

st.markdown("---")
# ==========================================================
# Input Section
# ==========================================================

from datetime import datetime

st.header("📝 Enter Trip Details")

left, right = st.columns(2)

with left:

    VendorID = st.selectbox(
        "Vendor ID",
        [1, 2]
    )

    passenger_count = st.number_input(
        "Passenger Count",
        min_value=1,
        max_value=9,
        value=1
    )

    trip_distance = st.number_input(
        "Trip Distance (Miles)",
        min_value=0.01,
        value=2.5,
        step=0.1
    )

    RatecodeID = st.selectbox(
        "Rate Code",
        [1, 2, 3, 4, 5, 99]
    )

    store_and_fwd_flag = st.selectbox(
        "Store & Forward",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    PULocationID = st.number_input(
        "Pickup Location ID",
        min_value=1,
        max_value=265,
        value=100
    )

    DOLocationID = st.number_input(
        "Dropoff Location ID",
        min_value=1,
        max_value=265,
        value=150
    )

with right:

    payment_options = {
        "Credit Card": 1,
        "Cash": 2,
        "No Charge": 3,
        "Dispute": 4,
        "Unknown": 5
    }

    payment_choice = st.selectbox(
        "Payment Type",
        list(payment_options.keys())
    )

    payment_type = payment_options[payment_choice]

    pickup_date = st.date_input(
        "Pickup Date",
        value=datetime.today()
    )

    from datetime import time

    pickup_time = st.time_input(
    "Pickup Time",
    value=time(12, 0)
)

    pickup_hour = pickup_time.hour
    pickup_day = pickup_date.day
    pickup_month = pickup_date.month
    pickup_weekday = pickup_date.weekday()

    trip_duration = st.number_input(
        "Trip Duration (Minutes)",
        min_value=1.0,
        value=15.0,
        step=1.0
    )

st.markdown("---")
# ==========================================================
# Prediction Section
# ==========================================================

if st.button("🚖 Predict Fare", use_container_width=True):

    # ----------------------------------------------------------
    # Validate Distance & Duration
    # ----------------------------------------------------------

    average_speed = trip_distance / (trip_duration / 60)

    if average_speed > 80:
        st.warning(
            f"⚠️ The entered distance and duration imply an average speed of {average_speed:.1f} mph. Please verify your inputs."
        )

    # ----------------------------------------------------------
    # Create Input DataFrame
    # ----------------------------------------------------------

    input_data = pd.DataFrame(
        [[
            VendorID,
            passenger_count,
            trip_distance,
            RatecodeID,
            store_and_fwd_flag,
            PULocationID,
            DOLocationID,
            payment_type,
            pickup_hour,
            pickup_day,
            pickup_month,
            pickup_weekday,
            trip_duration
        ]],
        columns=[
            "VendorID",
            "passenger_count",
            "trip_distance",
            "RatecodeID",
            "store_and_fwd_flag",
            "PULocationID",
            "DOLocationID",
            "payment_type",
            "pickup_hour",
            "pickup_day",
            "pickup_month",
            "pickup_weekday",
            "trip_duration"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.header("💰 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Fare",
            f"${prediction:.2f}"
        )

    with col2:
        st.metric(
            "Trip Distance",
            f"{trip_distance:.2f} miles"
        )

    st.success("Prediction generated successfully.")

    st.info(
        "💡 This fare is an estimate generated by the trained Random Forest model and may differ from the actual taxi fare."
    )
        # ======================================================
    # Prediction Explanation
    # ======================================================

    st.markdown("---")
    st.header("📝 Prediction Explanation")

    explanation = []

    if trip_distance >= 8:
        explanation.append(
            f"🚖 The trip distance of **{trip_distance:.2f} miles** is relatively long, which significantly increased the estimated fare."
        )
    elif trip_distance >= 3:
        explanation.append(
            f"🚖 The trip distance of **{trip_distance:.2f} miles** moderately influenced the estimated fare."
        )
    else:
        explanation.append(
            f"🚖 The short trip distance of **{trip_distance:.2f} miles** helped keep the fare lower."
        )

    if trip_duration >= 20:
        explanation.append(
            f"⏱️ The trip duration of **{trip_duration:.0f} minutes** contributed to a higher fare."
        )
    elif trip_duration <= 10:
        explanation.append(
            f"⏱️ The short trip duration of **{trip_duration:.0f} minutes** helped reduce the fare."
        )

    if pickup_hour in [7, 8, 9, 17, 18, 19]:
        explanation.append(
            "🚦 The selected pickup time falls during typical peak traffic hours."
        )

    if passenger_count >= 4:
        explanation.append(
            f"👥 Passenger count (**{passenger_count}**) had a small influence on the prediction."
        )

    top_feature = (
        feature_importance
        .sort_values(by="Importance", ascending=False)
        .iloc[0]["Feature"]
    )

    feature_names = {
        "VendorID": "Vendor ID",
        "passenger_count": "Passenger Count",
        "trip_distance": "Trip Distance",
        "RatecodeID": "Rate Code",
        "store_and_fwd_flag": "Store & Forward",
        "PULocationID": "Pickup Location",
        "DOLocationID": "Dropoff Location",
        "payment_type": "Payment Type",
        "pickup_hour": "Pickup Hour",
        "pickup_day": "Pickup Day",
        "pickup_month": "Pickup Month",
        "pickup_weekday": "Pickup Weekday",
        "trip_duration": "Trip Duration"
    }

    top_feature_name = feature_names.get(top_feature, top_feature)

    explanation.append(
        f"⭐ According to the trained Random Forest model, **{top_feature_name}** was the most influential feature."
    )

    for item in explanation:
        st.write(item)

    # ======================================================
    # Feature Importance
    # ======================================================

    st.markdown("---")
    st.header("📊 Feature Importance")

    feature_df = feature_importance.copy()

    feature_df["Feature"] = feature_df["Feature"].replace(feature_names)

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        feature_df["Feature"],
        feature_df["Importance"]
    )

    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Random Forest Feature Importance")

    st.pyplot(fig)

    st.subheader("📋 Feature Importance Scores")

    st.dataframe(
        feature_df.sort_values(
            by="Importance",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # Top 3 Most Important Features
    # ======================================================

    st.markdown("---")
    st.header("🏆 Top 3 Most Influential Features")

    medals = ["🥇", "🥈", "🥉"]

    top_features = feature_df.sort_values(
        by="Importance",
        ascending=False
    ).head(3)

    for medal, (_, row) in zip(medals, top_features.iterrows()):
        st.write(
            f"{medal} **{row['Feature']}** — Importance Score: **{row['Importance']:.4f}**"
        )

st.markdown("---")

st.caption(
    "🚖 Taxi Fare Prediction System | Built with Streamlit & Scikit-learn"
)