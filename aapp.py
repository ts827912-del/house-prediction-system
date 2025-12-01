import streamlit as st  
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ---------------------------
# 🔁 Load Model Function
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

model = load_model()

# ---------------------------
# 🧭 Sidebar Navigation
# ---------------------------
st.sidebar.title("🏠 Navigation")
page = st.sidebar.radio("Go to", ["Predict Price", "Model Details", "Visualizations"])

# ---------------------------
# 🔮 Page 1: Prediction Page
# ---------------------------
if page == "Predict Price":
    st.title("🏡 House Price Predictor")
    st.markdown("Enter house details below to predict the price.")

    # Input form
    area = st.number_input("Area (in sq ft)", min_value=300, max_value=10000, step=10)
    bedrooms = st.slider("Number of Bedrooms", 1, 10, 3)
    bathrooms = st.slider("Number of Bathrooms", 1, 5, 2)
    floors = st.slider("Number of Floors", 1, 4, 1)
    age = st.slider("Age of House (years)", 0, 100, 10)
    garage = st.slider("Garage Spaces", 0, 5, 1)
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    has_garden = st.radio("Has Garden?", ["Yes", "No"])
    has_basement = st.radio("Has Basement?", ["Yes", "No"])

    # Convert categorical inputs
    has_garden = 1 if has_garden == "Yes" else 0
    has_basement = 1 if has_basement == "Yes" else 0

    if st.button("Predict Price"):
        input_df = pd.DataFrame([{
            "Area": area,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Floors": floors,
            "Age": age,
            "Garage": garage,
            "Location": location,
            "HasGarden": has_garden,
            "HasBasement": has_basement
        }])

        # One-hot encode Location if needed
        if "Location" not in model.feature_names_in_:
            location_encoded = pd.get_dummies([location], prefix="Location").reindex(
                columns=["Location_Urban", "Location_Suburban", "Location_Rural"],
                fill_value=0
            )
            input_df = pd.concat([input_df.drop("Location", axis=1), location_encoded], axis=1)

        predicted_price = model.predict(input_df)[0]
        st.success(f"🏠 Estimated House Price: *${predicted_price:,.2f}*")

    # Sample prediction chart
    st.header("📈 Sample Prediction Distribution (Demo)")
    sample_data = pd.DataFrame({
        "Area": np.linspace(500, 5000, 50),
        "Bedrooms": [bedrooms]*50,
        "Bathrooms": [bathrooms]*50,
        "Floors": [floors]*50,
        "Age": [age]*50,
        "Garage": [garage]*50,
        "Location": [location]*50,
        "HasGarden": [has_garden]*50,
        "HasBasement": [has_basement]*50
    })

    if "Location" not in model.feature_names_in_:
        location_encoded = pd.get_dummies(sample_data["Location"], prefix="Location").reindex(
            columns=["Location_Urban", "Location_Suburban", "Location_Rural"],
            fill_value=0
        )
        sample_data = pd.concat([sample_data.drop("Location", axis=1), location_encoded], axis=1)

    pred_prices = model.predict(sample_data)
    st.line_chart({"Predicted Price": pred_prices})

# ---------------------------
# 📘 Page 2: Model Details
# ---------------------------
elif page == "Model Details":
    st.title("📘 Model Details")
    st.markdown("Here you can learn more about the model behind the predictions.")

    st.header("🧠 Model Overview")
    st.markdown("""
    This machine learning model predicts house prices based on:
    - Area (in square feet)
    - Bedrooms and bathrooms
    - Floors, garage spaces
    - Age of the property
    - Location type
    - Garden and basement features
    """)

    st.header("⚙️ Model Type")
    st.markdown("""
    The model is a **Random Forest Regressor** trained using scikit-learn.
    """)

    # ---------------------------
    # 📂 PROJECT DETAILS
    # ---------------------------
    st.header("📂 Project Overview")
    st.markdown("""
    This project is a **House Price Prediction System** built using machine learning.
    It helps users estimate house prices based on property features such as area,
    bedrooms, bathrooms, floors, location, and amenities.

    The system provides:
    - Price prediction using a trained ML model  
    - Data visualizations like heatmaps, scatter plots, and trends  
    - A user-friendly interface using Streamlit  
    """)

    st.header("🛠 Tools & Technologies Used")
    st.markdown("""
    - **Python** – Core programming language  
    - **Pandas** – Data cleaning and preprocessing  
    - **NumPy** – Numerical operations  
    - **Matplotlib & Seaborn** – Data visualization  
    - **Scikit-learn** – Machine learning model (Random Forest)  
    - **Joblib** – Saving and loading ML model  
    - **Streamlit** – Interactive web interface  
    """)

    st.header("✨ Advantages of the Project")
    st.markdown(""" 
    - ✔ Fast and accurate predictions  
    - ✔ Easy-to-use interface  
    - ✔ Interactive graphs  
    - ✔ Free and open-source tools  
    - ✔ Useful for buyers, sellers, agents  
    """)

    st.header("⚠️ Limitations & Disadvantages")
    st.markdown("""
    - ❌ Accuracy depends on dataset quality  
    - ❌ Cannot capture real-time market changes  
    - ❌ Requires consistent labels for categorical data  
    - ❌ Predictions vary if unseen data is provided  
    """)

    st.header("📌 Future Enhancements")
    st.markdown("""
    - 🔹 Add file upload for datasets  
    - 🔹 Use advanced models like XGBoost  
    - 🔹 Map-based visualizations  
    - 🔹 Deploy online (Streamlit Cloud / AWS)  
    - 🔹 Add login system  
    """)

# ---------------------------
# 📉 Page 3: Visualizations
# ---------------------------
elif page == "Visualizations":
    st.title("📊 Visualizations Dashboard")
    st.write("Explore housing data with correlation heatmaps and price trends.")

    @st.cache_data
    def load_data():
        try:
            return pd.read_csv("house_data.csv")
        except:
            return None

    df = load_data()

    if df is None:
        st.error("❌ Dataset file not found. Please place 'house_data.csv' in the same folder.")
    else:
        st.success("✅ Dataset loaded successfully!")

        # Correlation heatmap
        st.subheader("📌 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)

        # Scatter plot (recommended)
        st.subheader("📌 Price vs Area (Scatter Plot)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.scatter(df["Area"], df["Price"], alpha=0.5)
        ax3.set_xlabel("Area (sq ft)")
        ax3.set_ylabel("Price")
        ax3.set_title("Price vs Area")
        st.pyplot(fig3)
