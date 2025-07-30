import streamlit as st
import pandas as pd
from model import train_model
import matplotlib.pyplot as plt

st.title("💼 Employee Salary Predictor")

uploaded_file = st.file_uploader("Upload CSV File", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview", data.head())

    if 'YearsExperience' in data.columns and 'Salary' in data.columns:
        model = train_model(data)

        st.success("Model trained!")

        # Input prediction
        experience = st.slider("Years of Experience", 0.0, 20.0, 1.0, step=0.1)
        predicted_salary = model.predict([[experience]])[0]
        st.metric("Predicted Salary", f"${predicted_salary:,.2f}")

        # Visualization
        st.subheader("📈 Salary vs Experience")
        fig, ax = plt.subplots()
        ax.scatter(data['YearsExperience'], data['Salary'], color='blue', label='Data')
        ax.plot(data['YearsExperience'], model.predict(data[['YearsExperience']]), color='red', label='Model')
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Salary")
        ax.legend()
        st.pyplot(fig)
    else:
        st.error("CSV must have 'YearsExperience' and 'Salary' columns.")
