import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Graph Generator", layout="centered")
st.title("📊 Graph Generator")

# --- Data Source Selection ---
option = st.radio("Choose Data Input Method:", ("Upload CSV", "Enter Data Manually"))

if option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
elif option == "Enter Data Manually":
    st.write("### Enter your data:")
    default_data = pd.DataFrame({
        "X": [1, 2, 3, 4],
        "Y": [10, 20, 30, 40]
    })
    df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

# --- Plotting Section ---
if "df" in locals() and not df.empty:
    x_axis = st.selectbox("Select X-axis", df.columns, key="x_axis")
    y_axis = st.selectbox("Select Y-axis", df.columns, key="y_axis")

    x_label = st.text_input("X-axis Label", value=x_axis)
    y_label = st.text_input("Y-axis Label", value=y_axis)
    plot_title = st.text_input("Plot Title", value="My Line Plot")

    fig, ax = plt.subplots()
    ax.plot(df[x_axis], df[y_axis], marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(plot_title)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.info("Please upload a CSV or enter data to generate the graph.")
