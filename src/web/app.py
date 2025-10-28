"""Web streamlit"""
import streamlit as st
import requests
from src.config.loader import CONFIG

# --- Config ---
api_conf = CONFIG.get("api", {})
API_URL = f"http://{api_conf.get('host', '127.0.0.1')}:{api_conf.get('port', 8000)}"

# --- UI ---
st.set_page_config(page_title="Phishing Detection", page_icon="🔒")
st.title("Phishing Detection Web UI")
st.write("Enter a URL or an Email to check if it is potentially phishing.")

url_input = st.text_input("URL")
email_input = st.text_input("Email")

if st.button("Predict"):
    if not url_input and not email_input:
        st.warning("Please enter a URL or an Email.")
    else:
        payload = {"url": url_input, "email": email_input}
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            response.raise_for_status()
            data = response.json()
            st.success(f"Prediction: **{data['label']}**")
            st.info(f"Probability: {data['probability']:.2f}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error contacting API: {e}")
