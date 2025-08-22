import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load model & data
# -------------------------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("🔮 Employee Attrition Prediction Dashboard")

# -------------------------------
# Sidebar: Prediction Form
# -------------------------------
st.sidebar.header("📝 Employee Input")

# Example input fields (add more as needed)
age = st.sidebar.slider("Age", 18, 60, 30)
distance = st.sidebar.slider("DistanceFromHome", 1, 30, 10)
monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000, step=100)
business_travel = st.sidebar.selectbox("BusinessTravel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])


# Load model
model = joblib.load("best_rf.pkl")  # change to best_xgb.pkl if you want

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")  # dataset after preprocessing


data = load_data()


if st.sidebar.button("Predict"):
    # Prepare single-row input
    input_df = pd.DataFrame([{
        "Age": age,
        "DistanceFromHome": distance,
        "MonthlyIncome": monthly_income,
        "BusinessTravel": business_travel,
        "OverTime": overtime
    }])

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ This employee is **likely to leave** (Probability: {proba:.2f})")
    else:
        st.success(f"✅ This employee is **likely to stay** (Probability: {proba:.2f})")

# -------------------------------
# Dashboard Section
# -------------------------------
st.header("📊 Attrition Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Attrition Count")
    fig, ax = plt.subplots()
    sns.countplot(x="Attrition", data=data, ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Attrition by Department")
    fig, ax = plt.subplots()
    sns.countplot(x="Department", hue="Attrition", data=data, ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

# -------------------------------
# Feature Importance
# -------------------------------
st.header("⭐ Top Feature Importances")

# Extract feature names from pipeline
feature_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = model.named_steps['model'].feature_importances_

feat_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)

fig, ax = plt.subplots()
sns.barplot(x="Importance", y="Feature", data=feat_imp, ax=ax)
st.pyplot(fig)

# -------------------------------
# Filterable Table
# -------------------------------
st.header("🔎 High Risk Employees")
if "Attrition" in data.columns:
    high_risk = data[data["Attrition"] == 1].head(20)
    st.dataframe(high_risk)
else:
    st.warning("No 'Attrition' column found in dataset for filtering.")
