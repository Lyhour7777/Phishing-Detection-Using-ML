#!/usr/bin/env python3
"""
Streamlit Web Interface for Phishing Detection System.
"""
import streamlit as st
import requests
import sys
from pathlib import Path
from datetime import datetime

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# --- Page Configuration ---
st.set_page_config(
    page_title="Phishing Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/phishing-detection',
        'Report a bug': "https://github.com/yourusername/phishing-detection/issues",
        'About': "# Phishing Detection System\nAI-powered phishing URL detection"
    }
)

# --- Try to Load Config ---
try:
    from src.config.loader import CONFIG
    API_URL = f"http://{CONFIG.api.host}:{CONFIG.api.port}"
    APP_NAME = CONFIG.app.name
    MODEL_NAME = CONFIG.model.name
    config_loaded = True
except Exception as e:
    st.warning(f"⚠️ Config loading issue: {str(e)[:100]}... Using defaults.")
    API_URL = "http://127.0.0.1:8000"
    APP_NAME = "Phishing Detection System"
    MODEL_NAME = "distilbert-base-uncased"
    config_loaded = False

# --- Custom CSS ---
st.markdown("""
<style>
/* Detect Streamlit theme */
body[data-theme='dark'] .main-header {
    background: linear-gradient(135deg, #4a4a9a 0%, #5a3c80 100%);
    color: #f0f0f0;
}

body[data-theme='dark'] .result-box.safe {
    background-color: #225522;
    border: 2px solid #28a745;
    color: #d4f4d4;
}

body[data-theme='dark'] .result-box.phishing {
    background-color: #722222;
    border: 2px solid #dc3545;
    color: #f8d7da;
}

body[data-theme='dark'] .result-box.warning {
    background-color: #665500;
    border: 2px solid #ffc107;
    color: #fff7cc;
}

body[data-theme='dark'] .info-box {
    background-color: #2a3b50;
    border-left: 5px solid #2196F3;
    color: #e0e0e0;
}

body[data-theme='dark'] .stats-card {
    background-color: #333333;
    color: #f0f0f0;
}

body[data-theme='dark'] .url-display {
    background: #2a2a2a;
    border: 1px solid #555555;
    color: #ffffff;
}

/* Optional: buttons hover effect for dark mode */
body[data-theme='dark'] .stButton>button:hover {
    box-shadow: 0 4px 8px rgba(255, 255, 255, 0.2);
}

</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'example_urls' not in st.session_state:
    st.session_state.example_urls = {
        'safe': "https://google.com",
        'suspicious': "http://secure-bank-login.xyz",
        'phishing': "http://paypal-verify-account.top"
    }

# --- Helper Functions ---
def analyze_single_url(url, analyze_features=False):
    """Analyze a single URL and return results."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"url": url, "analyze_features": analyze_features},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store in history
            history_entry = {
                'url': url,
                'label': data.get('label', 'Unknown'),
                'probability': data.get('probability', 0),
                'is_phishing': data.get('is_phishing', False),
                'timestamp': datetime.now().isoformat(),
                'confidence': data.get('confidence', 'medium')
            }
            if history_entry not in st.session_state.analysis_history:
                st.session_state.analysis_history.append(history_entry)
            
            return data
        else:
            return {"error": f"API Error: {response.status_code}", "details": response.text[:100]}
            
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server", "details": f"Check if API is running at: {API_URL}"}
    except Exception as e:
        return {"error": f"Analysis failed", "details": str(e)[:100]}

def display_single_result(url, data):
    """Display single URL analysis results."""
    if "error" in data:
        st.error(f"❌ {data['error']}")
        if "details" in data:
            st.info(data['details'])
        return
    
    # Display results
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # URL Display
    st.markdown('<div class="url-display">🔗 ' + url + '</div>', unsafe_allow_html=True)
    
    # Result Box
    is_phishing = data.get('is_phishing', False)
    label = data.get('label', 'Unknown')
    probability = data.get('probability', 0)
    
    result_class = "phishing" if is_phishing else "safe"
    emoji = "⚠️" if is_phishing else "✅"
    
    st.markdown(f'<div class="result-box {result_class}">', unsafe_allow_html=True)
    
    # Header
    st.markdown(f"### {emoji} {label}")
    
    # Probability
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Confidence:** {probability:.1%}")
        st.progress(float(probability))
    with col2:
        st.metric("Risk", "High" if is_phishing else "Low")
    
    # Confidence
    confidence = data.get('confidence', 'medium').title()
    st.markdown(f"**Confidence Level:** {confidence}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Detailed Features
    if 'features' in data and data['features']:
        st.markdown("#### 🔍 Detailed Analysis")
        features = data['features']
        
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Security Indicators:**")
            if features.get('has_https'):
                st.success("✅ HTTPS Enabled")
            else:
                st.error("❌ No HTTPS")
            
            if not features.get('has_suspicious_keywords'):
                st.success("✅ No suspicious keywords")
            else:
                st.warning("⚠️ Suspicious keywords found")
            
            if features.get('tld_risk_score', 0) < 0.5:
                st.success("✅ Safe TLD")
            else:
                st.warning(f"⚠️ TLD Risk: {features.get('tld_risk_score', 0):.1%}")
        
        with cols[1]:
            st.markdown("**Technical Details:**")
            st.info(f"Domain Length: {features.get('domain_length', 0)} chars")
            st.info(f"Path Length: {features.get('path_length', 0)} chars")
            st.info(f"Subdomains: {'Yes' if features.get('has_subdomain') else 'No'}")
            st.info(f"Special Chars: {features.get('num_special_chars', 0)}")
    
    # Safety Tips
    st.markdown("---")
    if is_phishing:
        st.markdown("""
        <div class="info-box">
        <h4>⚠️ SECURITY WARNING</h4>
        <ul>
        <li><strong>DO NOT</strong> enter personal information</li>
        <li><strong>DO NOT</strong> click on links from this URL</li>
        <li><strong>DO</strong> report to your email provider</li>
        <li><strong>DO</strong> verify through official channels</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
        <h4>✅ Safety Recommendations</h4>
        <ul>
        <li>Always check for HTTPS in address bar</li>
        <li>Look for valid security certificates</li>
        <li>Be cautious with shortened URLs</li>
        <li>Verify domain spelling carefully</li>
        <li>When in doubt, don't proceed</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# --- Header ---
st.markdown(f'<div class="main-header"><h1>🔒 {APP_NAME}</h1></div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    
    # API Status
    st.markdown("#### 🔧 API Status")
    col1, col2 = st.columns(2)
    with col1:
        status_btn = st.button("🔄 Check", use_container_width=True, key="status_check")
    with col2:
        if st.button("📋 Docs", use_container_width=True, key="api_docs"):
            st.markdown(f"<script>window.open('{API_URL}/docs', '_blank');</script>", unsafe_allow_html=True)
    
    if status_btn:
        with st.spinner("Checking..."):
            try:
                resp = requests.get(f"{API_URL}/health", timeout=3)
                if resp.status_code == 200:
                    st.success("✅ API Online")
                    data = resp.json()
                    st.caption(f"Model: {data.get('model', 'Unknown')}")
                    st.caption(f"Version: {data.get('version', '1.0.0')}")
                else:
                    st.error(f"❌ API Error: {resp.status_code}")
            except Exception as e:
                st.error(f"❌ API Offline: {str(e)[:50]}")
    
    st.divider()
    
    # Quick Stats
    st.markdown("#### 📊 Quick Stats")
    if st.session_state.analysis_history:
        total = len(st.session_state.analysis_history)
        phishing = sum(1 for item in st.session_state.analysis_history if item.get('is_phishing'))
        safe = total - phishing
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", total)
        with col2:
            st.metric("Phishing", phishing, delta=f"{phishing/total*100:.1f}%" if total > 0 else "0%")
    else:
        st.info("No analyses yet")
    
    st.divider()
    
    # Configuration Info
    st.markdown("#### ⚙️ Configuration")
    st.caption(f"API: {API_URL}")
    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Config: {'✅ Loaded' if config_loaded else '⚠️ Defaults'}")
    
    if st.button("🧹 Clear History", type="secondary", key="clear_history"):
        st.session_state.analysis_history = []
        st.session_state.batch_results = None
        st.rerun()
    
    st.divider()
    
    # About
    st.markdown("#### ℹ️ About")
    st.caption("""
    This system uses machine learning to detect phishing URLs.
    
    **Features:**
    - Real-time URL analysis
    - Batch processing
    - Detailed threat insights
    
    **Note:** This is a demo system. Always verify through official channels.
    """)

# --- Main Content ---
tab1, tab2, tab3 = st.tabs(["🔍 Single URL", "📁 Batch Analysis", "📈 History & Stats"])

with tab1:
    st.markdown("### Single URL Analysis")
    
    # Example URLs section - placed BEFORE the text input
    with st.expander("💡 Example URLs", expanded=False):
        st.markdown("Try these example URLs:")
        
        col1, col2, col3 = st.columns(3)
        
        # Create buttons that set example URLs
        if col1.button("Safe Example", use_container_width=True, key="safe_example_btn"):
            st.session_state.example_selected = st.session_state.example_urls['safe']
        
        if col2.button("Suspicious Example", use_container_width=True, key="suspicious_example_btn"):
            st.session_state.example_selected = st.session_state.example_urls['suspicious']
        
        if col3.button("Phishing Example", use_container_width=True, key="phishing_example_btn"):
            st.session_state.example_selected = st.session_state.example_urls['phishing']
        
        # Show current example URLs
        st.markdown("**Current examples:**")
        st.code("\n".join([
            f"Safe: {st.session_state.example_urls['safe']}",
            f"Suspicious: {st.session_state.example_urls['suspicious']}",
            f"Phishing: {st.session_state.example_urls['phishing']}"
        ]))
    
    # Get initial value from session state if example was selected
    initial_url = ""
    if hasattr(st.session_state, 'example_selected'):
        initial_url = st.session_state.example_selected
        # Clear it after use
        del st.session_state.example_selected
    
    # URL input - placed AFTER the example section
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input(
            "Enter URL to analyze:",
            value=initial_url,  # Use the example if one was selected
            placeholder="https://example.com or example.com",
            help="Enter a complete URL including http:// or https://",
            key="single_url_input"  # Changed key from 'single_url'
        )
    with col2:
        analyze_features = st.checkbox("Detailed Analysis", help="Show detailed feature analysis", key="detailed_check")
    
    # Analyze button
    analyze_btn = st.button(
        "🔍 Analyze URL",
        type="primary",
        use_container_width=True,
        disabled=not url_input,
        key="analyze_btn"
    )
    
    if analyze_btn and url_input:
        with st.spinner(f"Analyzing `{url_input[:50]}...`"):
            result = analyze_single_url(url_input, analyze_features)
            display_single_result(url_input, result)

with tab2:
    st.markdown("### Batch URL Analysis")
    
    # Example batch URLs
    if 'batch_text' not in st.session_state:
        st.session_state.batch_text = ""
    
    batch_input = st.text_area(
        "Enter URLs (one per line):",
        value=st.session_state.batch_text,
        placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com",
        height=200,
        help="Enter multiple URLs to analyze at once",
        key="batch_input_area"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        analyze_batch = st.button(
            "📊 Analyze Batch",
            type="primary",
            use_container_width=True,
            disabled=not batch_input,
            key="analyze_batch_btn"
        )
    with col2:
        clear_batch = st.button("🗑️ Clear", type="secondary", use_container_width=True, key="clear_batch_btn")
    with col3:
        example_batch = st.button("📋 Load Example", type="secondary", use_container_width=True, key="example_batch_btn")
    
    if example_batch:
        example_urls = """https://google.com
            https://github.com
            http://secure-login-bank.xyz
            http://paypal-verification.click
            https://microsoft.com
            http://account-update-required.top"""
        st.session_state.batch_text = example_urls
        st.rerun()
    
    if clear_batch:
        st.session_state.batch_text = ""
        st.session_state.batch_results = None
        st.rerun()
    
    if analyze_batch and batch_input:
        urls = [url.strip() for url in batch_input.split('\n') if url.strip()]
        
        if not urls:
            st.warning("Please enter at least one URL")
            st.stop()
        
        if len(urls) > 50:
            st.warning(f"Maximum 50 URLs allowed. You entered {len(urls)}.")
            st.stop()
        
        with st.spinner(f"Analyzing {len(urls)} URLs..."):
            try:
                response = requests.post(
                    f"{API_URL}/batch_predict",
                    json={"urls": urls, "analyze_features": False},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.batch_results = data
                    
                    # Display summary
                    summary = data.get('summary', {})
                    st.markdown("## 📈 Batch Analysis Summary")
                    
                    # Stats Cards
                    cols = st.columns(4)
                    with cols[0]:
                        st.markdown('<div class="stats-card">Total<br><span class="stats-value">' + 
                                   str(summary.get('total_urls', 0)) + '</span></div>', unsafe_allow_html=True)
                    with cols[1]:
                        st.markdown('<div class="stats-card">Safe<br><span class="stats-value">' + 
                                   str(summary.get('safe_count', 0)) + '</span></div>', unsafe_allow_html=True)
                    with cols[2]:
                        st.markdown('<div class="stats-card">Phishing<br><span class="stats-value">' + 
                                   str(summary.get('phishing_count', 0)) + '</span></div>', unsafe_allow_html=True)
                    with cols[3]:
                        rate = summary.get('phishing_rate', 0)
                        st.markdown(f'<div class="stats-card">Phishing Rate<br><span class="stats-value">{rate:.1%}</span></div>', 
                                  unsafe_allow_html=True)
                    
                    # Detailed Results
                    st.markdown("### 🔍 Detailed Results")
                    predictions = data.get('predictions', [])
                    
                    for pred in predictions:
                        with st.expander(f"{'⚠️' if pred.get('is_phishing') else '✅'} {pred.get('url', 'Unknown')[:60]}...", 
                                       expanded=False):
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**URL:** `{pred.get('url')}`")
                                st.markdown(f"**Result:** **{pred.get('label')}**")
                            with col2:
                                prob = pred.get('probability', 0)
                                st.metric("Confidence", f"{prob:.1%}")
                            
                            # Add to history
                            history_entry = {
                                'url': pred.get('url'),
                                'label': pred.get('label'),
                                'probability': prob,
                                'is_phishing': pred.get('is_phishing', False),
                                'timestamp': datetime.now().isoformat(),
                                'confidence': pred.get('confidence', 'medium')
                            }
                            if history_entry not in st.session_state.analysis_history:
                                st.session_state.analysis_history.append(history_entry)
                
                else:
                    st.error(f"Batch analysis failed: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Batch analysis error: {str(e)}")
    
    # Display previous batch results
    elif st.session_state.batch_results:
        st.markdown("### Previous Batch Results")
        summary = st.session_state.batch_results.get('summary', {})
        st.info(f"Total URLs: {summary.get('total_urls', 0)} | "
                f"Phishing: {summary.get('phishing_count', 0)} | "
                f"Safe: {summary.get('safe_count', 0)}")

with tab3:
    st.markdown("### Analysis History & Statistics")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Analyze some URLs first!")
    else:
        # Statistics
        total = len(st.session_state.analysis_history)
        phishing = sum(1 for item in st.session_state.analysis_history if item.get('is_phishing'))
        safe = total - phishing
        avg_confidence = sum(item.get('probability', 0) for item in st.session_state.analysis_history) / total
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Analyses", total)
        with cols[1]:
            st.metric("Phishing", phishing, delta=f"{phishing/total*100:.1f}%")
        with cols[2]:
            st.metric("Safe", safe)
        with cols[3]:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # History Table
        st.markdown("### 📋 Recent Analyses")
        
        # Sort by timestamp (newest first)
        sorted_history = sorted(
            st.session_state.analysis_history, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        for i, item in enumerate(sorted_history[:20]):  # Show last 20
            with st.expander(f"{i+1}. {'⚠️' if item.get('is_phishing') else '✅'} {item.get('url', 'Unknown')[:50]}...", 
                           expanded=i < 3):
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**URL:** `{item.get('url')}`")
                    st.caption(f"Time: {item.get('timestamp', 'Unknown')}")
                with col2:
                    st.markdown(f"**Result:** **{item.get('label')}**")
                with col3:
                    prob = item.get('probability', 0)
                    st.progress(float(prob))
                    st.caption(f"{prob:.1%}")
        
        if total > 20:
            st.info(f"Showing 20 of {total} analyses. Clear history to remove old entries.")
        
        # Export option
        if st.button("💾 Export History", key="export_history"):
            import json
            import pandas as pd
            
            # Create DataFrame
            df = pd.DataFrame(st.session_state.analysis_history)
            
            # Offer download
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="phishing_analysis_history.csv",
                mime="text/csv",
                key="download_csv"
            )
            
            # Show preview
            st.markdown("**Preview:**")
            st.dataframe(df.head(), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 1rem;">
<p><strong>🔒 Phishing Detection System</strong> | Version 1.0.0</p>
<p>This tool assists in phishing detection but is not 100% accurate. Always verify through official channels.</p>
<p>Report phishing: <a href="https://safebrowsing.google.com/safebrowsing/report_phish/" target="_blank">Google Safe Browsing</a></p>
</div>
""", unsafe_allow_html=True)

# --- Auto-refresh warning ---
if not config_loaded:
    st.sidebar.warning("⚠️ Using default configuration. Check config loading.")