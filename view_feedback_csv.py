import streamlit as st
import pandas as pd

# Title
st.title("📋 Feedback Dataset Viewer")

# Load the CSV
try:
    df = pd.read_csv("feedback.csv")

    # Display the data table
    st.dataframe(df, use_container_width=True)

    # Optional: Show CSV download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, "feedback.csv", "text/csv")

except FileNotFoundError:
    st.error("❌ feedback.csv not found. Make sure it exists in the same directory as this script.")
