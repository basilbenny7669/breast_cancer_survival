import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# 1. Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = Path("models/breast_cancer_survival_rf.joblib")
    model = joblib.load(model_path)
    return model

model = load_model()

# -----------------------------
# 2. Helper: build input form
# -----------------------------
def get_user_input():
    st.sidebar.header("Patient details")

    # NOTE: Adjust min/max/defaults if you want
    age_at_diagnosis = st.sidebar.number_input(
        "Age at diagnosis (years)",
        min_value=18,
        max_value=100,
        value=55
    )

    tumor_size = st.sidebar.number_input(
        "Tumor size (mm)",
        min_value=0,
        max_value=200,
        value=20
    )

    lymph_nodes_examined_positive = st.sidebar.number_input(
        "Positive lymph nodes",
        min_value=0,
        max_value=50,
        value=1
    )

    # The options below should match (or be subset of) your dataset values.
    # After loading your CSV in a notebook, you can check e.g.:
    # df["type_of_breast_surgery"].unique()
    type_of_breast_surgery = st.sidebar.selectbox(
        "Type of breast surgery",
        options=[
            "BREAST CONSERVING",
            "MASTECTOMY",
        ]
    )

    cancer_type_detailed = st.sidebar.selectbox(
        "Cancer type (detailed)",
        options=[
            "Infiltrating Ductal Carcinoma",
            "Infiltrating Lobular Carcinoma",
            "Mucinous Carcinoma",
            "Other"
        ]
    )

    cellularity = st.sidebar.selectbox(
        "Cellularity",
        options=[
            "High",
            "Moderate",
            "Low"
        ]
    )

    er_status = st.sidebar.selectbox(
        "ER status",
        options=["Positive", "Negative"]
    )

    her2_status = st.sidebar.selectbox(
        "HER2 status",
        options=["Positive", "Negative"]
    )

    hormone_therapy = st.sidebar.selectbox(
        "Hormone therapy given?",
        options=["Yes", "No"]
    )

    chemotherapy = st.sidebar.selectbox(
        "Chemotherapy given?",
        options=["Yes", "No"]
    )

    neoplasm_histologic_grade = st.sidebar.selectbox(
        "Histologic grade",
        options=["1", "2", "3"]
    )

    # Build a single-row DataFrame.
    # Column names MUST match those used during training.
    data = {
        "age_at_diagnosis": age_at_diagnosis,
        "tumor_size": tumor_size,
        "lymph_nodes_examined_positive": lymph_nodes_examined_positive,
        "type_of_breast_surgery": type_of_breast_surgery,
        "cancer_type_detailed": cancer_type_detailed,
        "cellularity": cellularity,
        "er_status": er_status,
        "her2_status": her2_status,
        "hormone_therapy": hormone_therapy,
        "chemotherapy": chemotherapy,
        "neoplasm_histologic_grade": neoplasm_histologic_grade,
    }

    input_df = pd.DataFrame([data])
    return input_df


# -----------------------------
# 3. Main app
# -----------------------------
def main():
    st.set_page_config(
        page_title="Breast Cancer Survival Prediction",
        layout="wide"
    )

    st.title("Breast Cancer Survival Prediction")
    st.write(
        """
        This app uses a machine learning model trained on the METABRIC dataset 
        to **estimate overall survival** for breast cancer patients based on 
        clinical features.

        >  **Important:** This tool is for educational/demonstration purposes only 
        and must **not** be used for real medical decision-making.
        """
    )

    # Get user input from sidebar
    input_df = get_user_input()

    st.subheader("Entered patient details")
    st.dataframe(input_df)

    st.markdown("---")

    if st.button("Predict survival"):
        # Make prediction
        pred = model.predict(input_df)[0]

        # Try to get probability if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            # Assumes class 1 = survived, class 0 = not survived
            # But we should detect class index to be safe:
            if hasattr(model, "classes_"):
                classes_ = list(model.classes_)
                if 1 in classes_:
                    idx_alive = classes_.index(1)
                else:
                    idx_alive = 1 # fallback
            else:
                idx_alive = 1

            survival_prob = proba[idx_alive]
        else:
            survival_prob = None

        # Display result
        st.subheader("Prediction")

        if pred == 1:
            st.success(" Predicted outcome: **Survived**")
        else:
            st.error(" Predicted outcome: **Did not survive**")

        if survival_prob is not None:
            st.write(f"Estimated probability of survival: **{survival_prob:.1%}**")

        st.info(
            "These predictions are based on patterns learned from historical data "
            "and are intended only for learning and demonstration."
        )

    st.markdown("---")
    st.caption("Developed by Basil Benny: AI for Oncology")


if __name__ == "__main__":
    main()