import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_dashboard():
    st.title("Participant Feedback Analysis")

    # Load your feedback CSV
    df = pd.read_csv('feedback.csv')

    # Show basic stats
    st.write(f"Average Clarity Score: {df['Clarity Score'].mean():.2f}")
    st.write(f"Average Trust Score: {df['Trust Score'].mean():.2f}")
    st.write(f"Average UX Score: {df['UX Score'].mean():.2f}")

    # Plotting histograms
    for column in ['Clarity Score', 'Trust Score', 'UX Score']:
        st.subheader(f"{column} Distribution")
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=[1, 2, 3, 4, 5, 6], edgecolor='black', align='left')
        ax.set_xlabel('Score')
        ax.set_ylabel('Number of Participants')
        ax.set_xticks([1, 2, 3, 4, 5])
        st.pyplot(fig)

st.title("NeuroAid: Symptom Checker")

pid = st.text_input("Participant ID")
apply_clicked = st.button("Apply")
dashboard_clicked = st.button("Dashboard")

if apply_clicked:
    st.write(f"Participant ID applied: {pid}")
    # Your apply logic here

if dashboard_clicked:
    show_dashboard()
