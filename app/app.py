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

    payment_type = st.selectbox(
        "Payment Type",
        [1, 2, 3, 4, 5]
    )

    pickup_hour = st.slider(
        "Pickup Hour",
        0,
        23,
        12
    )

    pickup_day = st.slider(
        "Pickup Day",
        1,
        31,
        15
    )

    pickup_month = st.slider(
        "Pickup Month",
        1,
        12,
        1
    )

    pickup_weekday = st.slider(
        "Pickup Weekday",
        0,
        6,
        2
    )

    trip_duration = st.number_input(
        "Trip Duration (Minutes)",
        min_value=0.1,
        value=15.0,
        step=0.5
    )

st.markdown("---")
# ==========================================================
# Prediction Section
# ==========================================================

if st.button("🚖 Predict Fare", use_container_width=True):

    # Create input DataFrame
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

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.markdown("---")
    st.header("💰 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            label="Estimated Fare",
            value=f"${prediction:.2f}"
        )

    with result_col2:
        st.metric(
            label="Trip Distance",
            value=f"{trip_distance:.2f} miles"
        )

    st.success("Prediction generated successfully.")

    # ======================================================
    # Prediction Explanation
    # ======================================================

    st.markdown("---")
    st.header("📝 Prediction Explanation")

    explanation = []

    if trip_distance >= 8:
        explanation.append(
            "🚖 Long trip distance had the largest impact on increasing the predicted fare."
        )
    elif trip_distance >= 3:
        explanation.append(
            "🚖 Moderate trip distance contributed to the predicted fare."
        )
    else:
        explanation.append(
            "🚖 Short trip distance helped keep the predicted fare lower."
        )

    if trip_duration >= 20:
        explanation.append(
            "⏱️ Longer trip duration contributed to a higher fare."
        )
    elif trip_duration <= 10:
        explanation.append(
            "⏱️ Short trip duration helped reduce the fare."
        )

    if pickup_hour in [7, 8, 9, 17, 18, 19]:
        explanation.append(
            "🚦 The pickup time falls during peak traffic hours, which may slightly increase the fare."
        )

    if passenger_count >= 4:
        explanation.append(
            "👥 A larger passenger count had a minor influence on the prediction."
        )

    top_feature = (
        feature_importance
        .sort_values(by="Importance", ascending=False)
        .iloc[0]["Feature"]
    )

    explanation.append(
        f"⭐ According to the trained model, **{top_feature}** is the most influential feature."
    )

    for item in explanation:
        st.write(item)
    # ======================================================
    # Feature Importance
    # ======================================================

    st.markdown("---")
    st.header("📊 Feature Importance")

    feature_df = feature_importance.sort_values(
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
        )[["Feature", "Importance"]],
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # Top 3 Most Important Features
    # ======================================================

    st.markdown("---")
    st.header("🏆 Top 3 Most Influential Features")

    medals = ["🥇", "🥈", "🥉"]

    top_features = feature_importance.sort_values(
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
