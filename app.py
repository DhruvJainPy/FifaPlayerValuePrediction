import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# Load Trained Model & Data
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/xgboost_player_value.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_fifa_players.csv") 

model = load_model()
df = load_data()

# ----------------------------
# Streamlit App Layout
# ----------------------------
st.set_page_config(page_title="Football Player Market Value", layout="wide")

st.title("⚽ Football Player Market Value Prediction & Analysis")
st.markdown(
    "An interactive dashboard for **EDA, Model Evaluation, and Player Value Prediction** "
    "using a trained **XGBoost Regressor**."
)

# Create Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔍 EDA", "📈 Model Performance", "💰 Prediction"])

# ----------------------------
# 1. Data Overview
# ----------------------------
with tab1:
    st.header("📊 Dataset Overview")
    st.write(f"Entries: {df.shape[0]} | Features: {df.shape[1]}")
    st.dataframe(df.head(20))

    st.subheader("Statistical Summary")
    st.write(df.describe())

# ----------------------------
# 2. EDA
# ----------------------------
with tab2:
    st.header("🔍 Exploratory Data Analysis")

    st.subheader("Correlation Heatmap")
    st.image("assets/Correlation.png", caption="Correlation Matrix of Features", width='stretch')

    st.subheader("Distribution of Market Value (€)")
    fig, ax = plt.subplots()
    sns.histplot(df["value_euro"], bins=50, kde=True, ax=ax, color="skyblue")
    st.pyplot(fig)

    feature = st.selectbox("Select Feature to Compare with Market Value", df.select_dtypes(include=np.number).columns)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature], y=df["value_euro"], ax=ax, color="green")
    st.pyplot(fig)

# ----------------------------
# 3. Model Performance
# ----------------------------
with tab3:
    st.header("📈 Model Performance Evaluation")

    # Example metrics (replace with your actual ones)
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", "0.98")
    mse = 153591365.28
    rmse = np.sqrt(mse)

    # Format numbers
    mse_formatted = f"€{mse/1e6:.1f}M"  
    rmse_formatted = f"€{rmse/1e3:.1f}K" 

    col2.metric("MSE", mse_formatted)
    col3.metric("RMSE", rmse_formatted)

    st.subheader("Residuals Plot (Sample)")
    # Dummy example – replace with saved test predictions
    y_true = np.random.randint(1e5, 1e7, 100)
    y_pred = y_true + np.random.randint(-5e5, 5e5, 100)
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, color="orange")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Value (€)")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

# ----------------------------
# 4. Prediction
# ----------------------------
with tab4:
    st.header("💰 Predict Player Market Value")

    col1, col2 = st.columns(2)

    with col1:
        overall_rating = st.slider("Overall Rating", 40, 100, 75)
        age = st.slider("Age", 16, 45, 25)
        stamina = st.slider("Stamina", 0, 100, 70)
        balance = st.slider("Balance", 0, 100, 65)
        pace = st.slider("Pace", 0, 100, 70)
        passing = st.slider("Passing", 0, 100, 68)

    with col2:
        skills = st.slider("Skills", 0, 100, 60)
        mentality = st.slider("Mentality", 0, 100, 65)
        attacking_skills = st.slider("Attacking Skills", 0, 100, 72)
        def_strength = st.slider("Defensive Strength", 0, 100, 60)
        setpiece = st.slider("Setpiece Accuracy", 0, 100, 50)
        weak_foot = st.slider("Weak Foot (1-5)", 1, 5, 3)

        preferred_foot = st.selectbox("Preferred Foot", ["Left", "Right"])
        body_type = st.selectbox("Body Type", ["Lean", "Normal", "Stocky"])
        position = st.selectbox("Primary Position", ["Forward", "Midfielder", "Defender"])

    # Create input dataframe
    input_data = pd.DataFrame({
        "overall_rating": [overall_rating],
        "age": [age],
        "stamina": [stamina],
        "balance": [balance],
        "Pace": [pace],
        "Passing": [passing],
        "Skills": [skills],
        "Mentality": [mentality],
        "Attacking_skills": [attacking_skills],
        "Defensive_Strength": [def_strength],
        "Setpiece_accuracy": [setpiece],
        "weak_foot(1-5)": [weak_foot],
        "preferred_foot": [preferred_foot],
        "body_type": [body_type],
        "positions": [position]
    })

    # Preprocessing (One-hot encoding)
    input_data = pd.get_dummies(input_data)

    # Align with model features
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]

    # Predict
    if st.button("Predict Market Value (€)"):
        prediction = model.predict(input_data)
        st.success(f"💶 Predicted Market Value: €{prediction[0]:,.2f}")
