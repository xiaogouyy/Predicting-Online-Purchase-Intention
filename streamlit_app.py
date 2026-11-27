"""Streamlit frontend for online purchase intention prediction."""

import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Purchase Intention Explorer",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Purchase Intention Explorer")

st.write(
    "Upload online shopping session data to identify "
    "potential customers with high purchase intention."
)

# Required model input features
REQUIRED_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

# Sidebar settings
st.sidebar.header("Prediction Settings")

threshold = st.sidebar.slider(
    "Purchase probability threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

st.sidebar.caption(
    "A lower threshold targets more potential customers, "
    "while a higher threshold is more selective."
)

# Upload session data
st.header("1. Upload Session Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Validate required columns
        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )

        elif df.empty:
            st.error("The uploaded CSV contains no sessions.")

        else:
            st.success("Required columns are present.")

            # Keep only model input features
            features = df[REQUIRED_COLUMNS].copy()

            st.header("2. Data Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Sessions", len(features))
            col2.metric("Input Features", len(features.columns))
            col3.metric("Selected Threshold", f"{threshold:.0%}")

            st.dataframe(features.head(20))

            st.header("3. Purchase Predictions")

            # TODO: Connect to the FastAPI prediction endpoint.
            # TODO: Receive purchase probabilities.
            # TODO: Rank sessions by predicted probability.
            # TODO: Apply the selected threshold.
            # TODO: Enable CSV export of prediction results.

            st.info(
                "Prediction API integration is pending. "
                "Purchase probabilities will appear here "
                "once the backend is connected."
            )

    except Exception as e:
        st.error(f"Unable to read the CSV file: {e}")

else:
    st.info("Upload a session CSV file to get started.")

st.divider()

st.caption(
    "INSY 674 | Online Purchase Intention Prediction | "
    "Initial frontend prototype for offline analysis"
)
