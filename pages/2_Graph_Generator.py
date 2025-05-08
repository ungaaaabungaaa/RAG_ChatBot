import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📊 Graph Generator")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    x_axis = st.selectbox("Select X-axis", df.columns)
    y_axis = st.selectbox("Select Y-axis", df.columns)

    fig, ax = plt.subplots()
    ax.plot(df[x_axis], df[y_axis])
    st.pyplot(fig)
