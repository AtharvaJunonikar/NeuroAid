import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


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

st.markdown("---")
st.subheader("📂 View Complete Dataset")

if st.button("Feedback Dataset Viewer"):
    dataset_script = "view_feedback_csv.py"
    dataset_port = "8507"

    subprocess.Popen(["streamlit", "run", dataset_script, "--server.port", str(dataset_port)])

    dev_ip = get_local_ip()
    dataset_url = f"http://{dev_ip}:{dataset_port}"
    st.success(f"📊 Dataset Viewer started at: [Click to open it here]({dataset_url})")



# ✅ Call the function so it actually runs
if __name__ == "__main__":
    show_dashboard()
