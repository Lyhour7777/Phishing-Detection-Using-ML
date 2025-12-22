#!/usr/bin/env python3
"""
Streamlit Web Interface for Phishing Detection System (Updated for new API)
"""
import streamlit as st
import requests
from datetime import datetime

# --- Configuration ---
API_URL = "http://127.0.0.1:8000"  # Update if using a config file
APP_NAME = "Phishing Detection System"

# --- Page Config ---
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State ---
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

# --- Helper Functions ---
def analyze_single_url(url: str):
    """Call /predict endpoint for a single URL."""
    try:
        response = requests.post(f"{API_URL}/predict", json={"url": url}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text[:100]}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)[:100]}"}

def analyze_batch_urls(urls: list):
    """Call /batch_predict endpoint for multiple URLs."""
    try:
        response = requests.post(f"{API_URL}/batch_predict", json={"urls": urls}, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error {response.status_code}: {response.text[:100]}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)[:100]}"}

def display_single_result(url: str, data: dict):
    """Display single URL analysis."""
    if "error" in data:
        st.error(f"❌ {data['error']}")
        return
    
    label = data.get("label", "unknown")
    probability = data.get("probability", 0)
    is_phishing = label.lower() == "phishing"

    st.markdown("---")
    st.markdown(f"### 🔗 {url}")
    st.markdown(f"**Result:** {'⚠️ Phishing' if is_phishing else '✅ Legitimate'}")
    st.progress(float(probability))
    st.markdown(f"**Confidence:** {probability:.1%}")
    st.markdown("---")

# --- Header ---
st.markdown(f"# 🔒 {APP_NAME}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Single URL", "📁 Batch Analysis", "📈 History & Stats"])

# --- Single URL Tab ---
with tab1:
    url_input = st.text_input("Enter URL to analyze:", placeholder="https://example.com")
    analyze_btn = st.button("🔍 Analyze URL", disabled=not url_input)

    if analyze_btn and url_input:
        result = analyze_single_url(url_input.strip())
        display_single_result(url_input.strip(), result)
        if "error" not in result:
            st.session_state.analysis_history.append({
                "url": url_input.strip(),
                "label": result.get("label", "unknown"),
                "probability": result.get("probability", 0),
                "timestamp": datetime.now().isoformat()
            })

# --- Batch Analysis Tab ---
with tab2:
    batch_input = st.text_area("Enter URLs (one per line):", height=200)
    analyze_batch_btn = st.button("📊 Analyze Batch")
    clear_batch_btn = st.button("🗑️ Clear Batch Input")

    if clear_batch_btn:
        st.session_state.batch_results = None
        st.session_state.batch_text = ""
        batch_input = ""

    if analyze_batch_btn and batch_input:
        urls = [url.strip() for url in batch_input.split("\n") if url.strip()]
        if not urls:
            st.warning("Please enter at least one URL.")
        else:
            result = analyze_batch_urls(urls)
            st.session_state.batch_results = result
            if "error" in result:
                st.error(result["error"])
            else:
                for pred in result.get("predictions", []):
                    st.markdown(f"{pred['url']} -> {pred['label']} ({pred['probability']:.1%})")
                    st.session_state.analysis_history.append({
                        "url": pred['url'],
                        "label": pred['label'],
                        "probability": pred['probability'],
                        "timestamp": datetime.now().isoformat()
                    })

# --- History & Stats Tab ---
with tab3:
    history = st.session_state.analysis_history
    if not history:
        st.info("No analysis history yet.")
    else:
        total = len(history)
        phishing_count = sum(1 for x in history if x["label"].lower() == "phishing")
        safe_count = total - phishing_count
        avg_conf = sum(x["probability"] for x in history) / total

        st.metric("Total Analyses", total)
        st.metric("Phishing", phishing_count)
        st.metric("Legitimate", safe_count)
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

        st.markdown("### Recent Analyses")
        for i, item in enumerate(reversed(history[-20:])):
            st.markdown(f"{i+1}. {item['url']} -> {item['label']} ({item['probability']:.1%})")