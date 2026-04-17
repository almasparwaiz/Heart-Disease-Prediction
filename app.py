import streamlit as st
import pandas as pd
import joblib

# --- Load the Model, Scaler, and Feature Columns ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load("stacking_classifier_model.joblib")
        scaler = joblib.load("scaler.joblib")
        feature_columns = joblib.load("feature_columns.joblib")
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Model files not found. Make sure all .joblib files are in the same folder as app.py")
        st.stop()

model, scaler, feature_columns = load_resources()

# ==================================================
# PAGE SETTINGS
# ==================================================
st.set_page_config(
    page_title="Heart Risk Prediction App",
    page_icon="❤️",
    layout="centered"
)

# ==================================================
# TITLE
# ==================================================
st.title("❤️ Heart Risk Prediction")
st.write("Enter patient details below.")

# ==================================================
# SIDEBAR INPUT
# ==================================================
st.sidebar.header("Patient Information")

def yes_no(label):
    return st.sidebar.selectbox(
        label,
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

def gender_box():
    return st.sidebar.selectbox(
        "Gender",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

def user_input_features():

    data = {
        "Chest_Pain": yes_no("Chest Pain"),
        "Shortness_of_Breath": yes_no("Shortness of Breath"),
        "Fatigue": yes_no("Fatigue"),
        "Palpitations": yes_no("Palpitations"),
        "Dizziness": yes_no("Dizziness"),
        "Swelling": yes_no("Swelling"),
        "Pain_Arms_Jaw_Back": yes_no("Pain in Arms/Jaw/Back"),
        "Cold_Sweats_Nausea": yes_no("Cold Sweats / Nausea"),
        "High_BP": yes_no("High Blood Pressure"),
        "High_Cholesterol": yes_no("High Cholesterol"),
        "Diabetes": yes_no("Diabetes"),
        "Smoking": yes_no("Smoking"),
        "Obesity": yes_no("Obesity"),
        "Sedentary_Lifestyle": yes_no("Sedentary Lifestyle"),
        "Family_History": yes_no("Family History"),
        "Chronic_Stress": yes_no("Chronic Stress"),
        "Gender": gender_box(),
        "Age": st.sidebar.slider("Age", 20, 90, 45)
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ==================================================
# FEATURE ENGINEERING
# ==================================================
def apply_feature_engineering(df):
    df = df.copy()

    bins = [0, 40, 60, 100]
    labels = ["Young", "Middle_Aged", "Elderly"]
    df["Age_Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    risk_cols = [
        "High_BP", "High_Cholesterol", "Diabetes",
        "Smoking", "Obesity", "Family_History", "Chronic_Stress"
    ]

    df["Total_Risk_Factor_Score"] = df[risk_cols].sum(axis=1)

    heart_cols = [
        "Chest_Pain", "Shortness_of_Breath", "Fatigue",
        "Palpitations", "Dizziness", "Swelling",
        "Pain_Arms_Jaw_Back", "Cold_Sweats_Nausea"
    ]

    df["Heart_Workload_Index"] = df[heart_cols].sum(axis=1)

    df["High_BP_x_High_Cholesterol"] = df["High_BP"] * df["High_Cholesterol"]
    df["Age_x_High_Cholesterol"] = df["Age"] * df["High_Cholesterol"]
    df["Diabetes_x_Obesity"] = df["Diabetes"] * df["Obesity"]

    return df

engineered_df = apply_feature_engineering(input_df)

# ==================================================
# MATCH TRAINING COLUMNS
# ==================================================
final_input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

for col in final_input_df.columns:
    if col in engineered_df.columns:
        final_input_df[col] = engineered_df[col]

# One-hot Age Group
age_group = engineered_df["Age_Group"].iloc[0]

if "Age_Category_Middle_Aged" in final_input_df.columns:
    final_input_df["Age_Category_Middle_Aged"] = 1 if age_group == "Middle_Aged" else 0

if "Age_Category_Elderly" in final_input_df.columns:
    final_input_df["Age_Category_Elderly"] = 1 if age_group == "Elderly" else 0

# ==================================================
# SCALE
# ==================================================
scaled_input = scaler.transform(final_input_df)

# ==================================================
# PREDICTION
# ==================================================
if st.button("Predict Heart Risk"):

    prob = model.predict_proba(scaled_input)[:, 1][0]
    pred = 1 if prob > 0.5 else 0

    if pred == 1:
        st.error(f"⚠ HIGH RISK Detected ({prob:.2%})")
    else:
        st.success(f"✅ LOW RISK ({prob:.2%})")
