import os
import sys

# Add the project root and 'src' directory to sys.path to resolve ModuleNotFoundError
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import time
import streamlit as st
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx
import numpy as np
from PIL import Image
import uuid
import hashlib
from pathlib import Path
from cnnClassifier.pipeline.prediction import PredictionPipeline
import io
import base64
import pandas as pd
from groq import Groq
try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import plotly.express as px
except ImportError:
    px = None

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress extraneous TensorFlow console logs

import warnings
warnings.filterwarnings("ignore")

# --- Application Page Configuration: Set up the Streamlit page title, icon, and layout ---
st.set_page_config(
    page_title="Renal Vision",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initial Model Download if missing ---
@st.cache_resource(show_spinner="Downloading Kidney Disease Classification Model from Google Drive. Please wait...")
def download_model_if_missing():
    from pathlib import Path
    import shutil

    os.makedirs("model", exist_ok=True)

    downloaded_onnx = Path("model/model.onnx")
    local_onnx = Path("artifacts/training/model.onnx")
    if not downloaded_onnx.exists() and local_onnx.exists():
        shutil.copy(local_onnx, downloaded_onnx)

    downloaded_h5 = Path("model/model.h5")
    local_h5 = Path("artifacts/training/model.h5")
    if not downloaded_h5.exists() and local_h5.exists():
        shutil.copy(local_h5, downloaded_h5)

    try:
        import gdown
    except ImportError:
        st.error("Please install gdown (`pip install gdown`) to download the model automatically.")
        return
    
    # Using the provided Drive link for all predictions and visualizations
    onnx_file_id = "1FNW-B0dBBVwOAfrl6M5zGiNhqXVSYo75"
    h5_file_id = "1qHpMvmT68BCBpZB_DetH88Ob6KNI42Es"
    h5_file_id = "1lR8ZnfcNNVYqUczNQDVschTouSdhCg1R"

    
    # Check if file exists but is just a tiny Git LFS pointer or HTML error page (< 1MB)
    if downloaded_onnx.exists() and os.path.getsize(downloaded_onnx) < 1000:
        pass # Do not automatically remove, rely on user to provide correct model

    if not downloaded_onnx.exists():
        try:
            gdown.download(id=onnx_file_id, output=str(downloaded_onnx), quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(id=onnx_file_id, output=str(downloaded_onnx), quiet=False)

    # Ensure the original Keras model is also downloaded for XAI visualizations (Grad-CAM/Attention)
    
    if downloaded_h5.exists() and os.path.getsize(downloaded_h5) < 1000:
        pass # Do not automatically remove, rely on user to provide correct model

    if not downloaded_h5.exists():
        try:
            gdown.download(id=h5_file_id, output=str(downloaded_h5), quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(id=h5_file_id, output=str(downloaded_h5), quiet=False)

download_model_if_missing()

ONNX_MODEL_PATH = "model/model.onnx"

# --- Initialize Streamlit Session State Variables: Ensure all required state variables are defined to maintain state across reruns ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'latest_label' not in st.session_state:
    st.session_state.latest_label = None
if 'latest_confidence' not in st.session_state:
    st.session_state.latest_confidence = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_clear_confirm' not in st.session_state:
    st.session_state.show_clear_confirm = False
if 'show_clear_scan_confirm' not in st.session_state:
    st.session_state.show_clear_scan_confirm = False
if 'show_clear_batch_confirm' not in st.session_state:
    st.session_state.show_clear_batch_confirm = False
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'
if 'processed_file_id' not in st.session_state:
    st.session_state.processed_file_id = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = None
if 'batch_pdf_bytes' not in st.session_state:
    st.session_state.batch_pdf_bytes = None
if 'batch_csv_bytes' not in st.session_state:
    st.session_state.batch_csv_bytes = None
if 'batch_file_ids' not in st.session_state:
    st.session_state.batch_file_ids = []
if 'prediction_active' not in st.session_state:
    st.session_state.prediction_active = False
if 'persisted_label' not in st.session_state:
    st.session_state.persisted_label = None
if 'persisted_confidence' not in st.session_state:
    st.session_state.persisted_confidence = None
if 'persisted_heatmap' not in st.session_state:
    st.session_state.persisted_heatmap = None
if 'persisted_attention' not in st.session_state:
    st.session_state.persisted_attention = None
if 'persisted_lime' not in st.session_state:
    st.session_state.persisted_lime = None
if 'persisted_img' not in st.session_state:
    st.session_state.persisted_img = None
if 'persisted_preprocessed_img' not in st.session_state:
    st.session_state.persisted_preprocessed_img = None
if 'persisted_file_bytes' not in st.session_state:
    st.session_state.persisted_file_bytes = None
if 'persisted_file_ext' not in st.session_state:
    st.session_state.persisted_file_ext = None
if 'generating_attention' not in st.session_state:
    st.session_state.generating_attention = False
if 'generating_lime' not in st.session_state:
    st.session_state.generating_lime = False
if 'persisted_color_indicator' not in st.session_state:
    st.session_state.persisted_color_indicator = None
if 'persisted_animation_class' not in st.session_state:
    st.session_state.persisted_animation_class = None
if 'persisted_probabilities' not in st.session_state:
    st.session_state.persisted_probabilities = None
if 'batch_processing_running' not in st.session_state:
    st.session_state.batch_processing_running = False
if 'batch_progress' not in st.session_state:
    st.session_state.batch_progress = 0.0
if 'batch_results_temp' not in st.session_state:
    st.session_state.batch_results_temp = None
if 'batch_status_message' not in st.session_state:
    st.session_state.batch_status_message = ""
if 'batch_prediction_cache' not in st.session_state:
    st.session_state.batch_prediction_cache = {}
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = None
if 'batch_uploader_key' not in st.session_state:
    st.session_state.batch_uploader_key = 0
if 'single_uploader_key' not in st.session_state:
    st.session_state.single_uploader_key = 0

# --- Inject Custom Medical Theme CSS: Apply custom styling based on the active theme (light or dark) ---
if st.session_state.theme == 'dark':
    bg_color = "#000000"           # Pure Black Background
    grid_color = "rgba(255, 255, 255, 0.06)" # Distinct white grid for structured look
    text_color = "#F8FAFC"         # Crisp Off-White Text
    container_bg = "#0A0A0A"       # Deep Contrast Container
    container_border = "#2A2A2A"   # Sharp Dark Border
    sidebar_bg = "#FFFFFF"         # Pure White for Sidebar
    sidebar_border = "#E2E8F0"     # Sharp Light Border
    sidebar_text = "#1E282D"       # Dark Text for White Sidebar
    assistant_text = "#D1FAE5"     # Soft Emerald Text for AI
    title_color = "#38BDF8"        # Bright Cyan for Headers
    main_title_shadow = "0px 2px 8px rgba(56, 189, 248, 0.25)" # Subtle cyan glow for main title
    val_color = "#E0F2FE"          # Very Light Blue for Values
    info_bg = "rgba(56, 189, 248, 0.3)"     # Soft Cyan for Info
    success_bg = "rgba(80, 200, 120, 0.3)"  # Soft Green for Success
    warning_bg = "rgba(217, 83, 79, 0.3)"   # Soft Red for Warning
    expander_bg = "#1E293B"                 # Slate Blue for Dark Mode Expander
    expander_border = "#334155"             # Subtle Dark Border
    expander_text = "#F8FAFC"               # Crisp Off-White Text
    footer_bg = "#FFFFFF"                   # White Footer Background for Dark Theme
    footer_text = "#000000"                 # Black Text for White Footer
    divider_color = "#FFFFFF"               # White Dividers for Dark Theme
    user_chat_bg = "#1E293B"                # Deep Slate Blue for User in Dark Theme
    assistant_chat_bg = "rgba(0, 123, 255, 0.15)" # Clinical Medical Blue for Assistant in Dark Theme
    header_shadow = "0 4px 12px rgba(0, 0, 0, 0.4)" # Deeper shadow for the light header on dark bg
    header_bg_color = "#F8F9FA"             # Off-white header to distinguish from pure white sidebar
    header_border_color = "#E2E8F0"         # Light border for the off-white header
    header_text_color = "#000000"           # Black text for white header
    header_subtitle_color = "#6C757D"       # Grey subtitle for white header
    scrollbar_color = "rgba(255, 255, 255, 0.25)"
    scrollbar_hover = "rgba(255, 255, 255, 0.5)"
    sidebar_scrollbar_color = "rgba(0, 0, 0, 0.2)"  # Dark scrollbar for the white sidebar
    sidebar_scrollbar_hover = "rgba(0, 0, 0, 0.4)"
    sidebar_icon_color = "#1E282D"          # Dark slate for white sidebar
    sidebar_icon_bg = "rgba(56, 189, 248, 0.25)" # Bright cyan button background for visibility
    chat_input_bg = "#1E293B"               # Deep Slate Blue for Chat Input
    chat_input_text_color = "#F8FAFC"       # Crisp Off-White Text for Chat Input
    chat_btn_hover_bg = "#007BFF"           # Medical Blue for Chat Send Button Hover
    chat_btn_hover_color = "#FFFFFF"        # White Icon for Chat Send Button Hover
    sidebar_toggle_bg = "#101010"                   # black to stand out on white header bg
    sidebar_toggle_border = "#38BDF8"               # Cyan border for accent
    sidebar_toggle_icon = "#000000"                 # Black icon for high contrast
    sidebar_toggle_hover_bg = "#3D9DDD"             # Subtle blue tint on hover
    sidebar_toggle_shadow = "0 4px 15px rgba(56, 189, 248, 0.4)"
    sidebar_toggle_pulse = "rgba(56, 189, 248, 0.6)" # Keep cyan pulse
    chat_scrollbar_color = "rgba(56, 189, 248, 0.3)"    # Cyan scrollbar for chat
    chat_scrollbar_hover = "rgba(56, 189, 248, 0.6)"
else:
    bg_color = "#FFFFFF"           # Pure White Background
    grid_color = "#E2E8F0"         # Solid light gray grid instead of transparent
    text_color = "#0C0E11"         # Rich Dark Slate for maximum readability
    container_bg = "#FFFFFF"       # Solid white container for maximum readability
    container_border = "#D1D5DB"   # Solid light gray border
    sidebar_bg = "#FFFFFF"         # Pure White Sidebar
    sidebar_border = "#E2E8F0"     # Sharp Light Border
    sidebar_text = "#1E282D"       # Dark Slate Text
    assistant_text = "#065F46"     # Deep Forest Green Text for AI
    title_color = "#1E282D"        # Clinical Cerulean Blue Headers
    main_title_shadow = "0px 2px 5px rgba(0, 0, 0, 0.15)" # Subtle dark shadow for main title
    val_color = "#4B6878"          # Deep Medical Blue for Values
    info_bg = "#E0F2FE"            # Solid Soft Blue for Info
    success_bg = "#DCFCE7"         # Solid Soft Green for Success
    warning_bg = "#FEE2E2"         # Solid Soft Red for Warning
    expander_bg = "#AEC6CF"                 # Pastel Blue for Expander
    expander_border = "#779ECB"             # Darker Pastel Blue Border
    expander_text = "#0F172A"               # Dark Slate Text
    footer_bg = "#000000"                   # Black Footer Background for Light Theme
    footer_text = "#FFFFFF"                 # White Text for Black Footer
    divider_color = "#000000"               # Black Dividers for Light Theme
    user_chat_bg = "#F1F5F9"       # Clean Clinical Slate for User Chat
    assistant_chat_bg = "#E6F0FD"  # Soft Medical Blue for Assistant Chat
    header_shadow = "0 4px 12px rgba(0, 0, 0, 0.08)" # Softer shadow for the dark header on light bg
    header_bg_color = "#000000"             # Solid black header for Light Theme
    header_border_color = "#333333"         # Dark border for the black header
    header_text_color = "#FFFFFF"           # White text for black header
    header_subtitle_color = "#CBD5E1"       # Light grey subtitle for black header
    scrollbar_color = "#CBD5E1"    # Solid slate scrollbar
    scrollbar_hover = "#94A3B8"    # Solid slate hover
    sidebar_scrollbar_color = "#CBD5E1"
    sidebar_scrollbar_hover = "#94A3B8"
    sidebar_icon_color = "#1E282D"          # Dark slate
    sidebar_icon_bg = "#F1F5F9"    # Solid soft grey background
    chat_input_bg = "#000000"               # Black for Chat Input in Light Theme
    chat_input_text_color = "#FFFFFF"       # White text for Chat Input in Light Theme
    chat_btn_hover_bg = "#0056b3"           # Darker Blue for Chat Send Button Hover
    chat_btn_hover_color = "#FFFFFF"        # White Icon for Chat Send Button Hover
    sidebar_toggle_bg = "#007BFF"           # Vibrant blue for toggle
    sidebar_toggle_border = "#0056b3"       # Darker blue border
    sidebar_toggle_icon = "#FFFFFF"         # Crisp white icon
    sidebar_toggle_hover_bg = "#0056b3"     # Darker blue on hover
    sidebar_toggle_shadow = "0 4px 15px rgba(0, 123, 255, 0.4)"
    sidebar_toggle_pulse = "rgba(0, 123, 255, 0.6)"     # Blue pulse for the white theme
    chat_scrollbar_color = "rgba(0, 123, 255, 0.3)"     # Blue scrollbar for chat
    chat_scrollbar_hover = "rgba(0, 123, 255, 0.6)"

st.markdown(f"""
    <style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Slab:wght@500;700;800&display=swap');
    
    html, body {{
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
    }}

    /* Chat Input Submit Button Hover State */
    [data-testid="stChatInput"] button:hover {{
        background-color: {chat_btn_hover_bg} !important;
        color: {chat_btn_hover_color} !important;
        transform: scale(1.1) !important;
        transition: all 0.2s ease-in-out !important;
    }}
    [data-testid="stChatInput"] button:hover svg {{
        fill: {chat_btn_hover_color} !important;
        color: {chat_btn_hover_color} !important;
    }}

    /* Safely apply modern font to typography without breaking Streamlit's native icon ligatures (like the green tick) */
    p, label, .stMarkdown, .stText, [data-testid="stMetricValue"], summary, input, textarea, button {{
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Roboto Slab', serif !important;
    }}

    /* Custom Modern Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px !important;
        height: 6px !important;
    }}
    ::-webkit-scrollbar-track {{
        background: transparent !important;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: {scrollbar_color} !important;
        border-radius: 10px !important;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background-color: {scrollbar_hover} !important;
    }}
    html, body, [data-testid="stAppViewContainer"] {{
        scrollbar-width: thin;
        scrollbar-color: {scrollbar_color} transparent;
    }}

    /* Sidebar-specific Scrollbar for White Sidebar */
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb {{
        background-color: {sidebar_scrollbar_color} !important;
    }}
    [data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {{
        background-color: {sidebar_scrollbar_hover} !important;
    }}
    [data-testid="stSidebar"], [data-testid="stSidebarUserContent"] {{
        scrollbar-color: {sidebar_scrollbar_color} transparent !important;
    }}
    
    /* Specific Scrollbar for the Chat Container */
    div.element-container:has(#chat-scroll-target) + div ::-webkit-scrollbar-thumb {{
        background-color: {chat_scrollbar_color} !important;
    }}
    div.element-container:has(#chat-scroll-target) + div ::-webkit-scrollbar-thumb:hover {{
        background-color: {chat_scrollbar_hover} !important;
    }}
    div.element-container:has(#chat-scroll-target) + div, div.element-container:has(#chat-scroll-target) + div [data-testid="stVerticalBlock"] {{
        scrollbar-color: {chat_scrollbar_color} transparent !important;
    }}

    /* Base Streamlit App overrides */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
        background-image: 
            linear-gradient({grid_color} 1px, transparent 1px),
            linear-gradient(90deg, {grid_color} 1px, transparent 1px);
        background-size: 40px 40px;
        background-position: center center;
        color: {text_color};
    }}
    /* Force all dividers to match theme color */
    hr {{
        border-bottom: 2px dashed {divider_color} !important;
    }}
    /* Modern Medical Dashboard Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg} !important;
        background-image: linear-gradient(180deg, {sidebar_bg} 70%, rgba(0, 123, 255, 0.05) 100%) !important;
        border-right: 1px solid {sidebar_border} !important;
        box-shadow: 8px 0 24px rgba(0, 123, 255, 0.12) !important;
        transition: transform 0.4s ease-in-out, box-shadow 0.4s ease-in-out !important;
    }}
    [data-testid="stSidebarResizer"] {{
        background-color: {sidebar_border} !important;
        width: 3px !important;
        opacity: 1 !important;
        transition: background-color 0.3s ease !important;
    }}
    [data-testid="stSidebarResizer"]:hover {{
        background-color: #007BFF !important; /* Blue on hover */
    }}
    [data-testid="stSidebar"] .stMarkdownContainer p, [data-testid="stSidebar"] .stMarkdownContainer div {{
        color: {sidebar_text};
    }}
    [data-testid="stSidebarUserContent"] {{
        padding-top: 2rem !important;
    }}
    
    /* Sidebar Toggle Pulse Animation to attract attention on mobile */
    @keyframes pulseSidebarToggle {{
        0% {{ box-shadow: 0 0 0 0 {sidebar_toggle_pulse}; }}
        70% {{ box-shadow: 0 0 0 12px transparent; }}
        100% {{ box-shadow: 0 0 0 0 transparent; }}
    }}

    /* Ensure Sidebar Toggle Icons are clearly visible and distinguishable */
    [data-testid="collapsedControl"] {{
        background-color: {sidebar_toggle_bg} !important;
        border: 2px solid {sidebar_toggle_border} !important;
        border-radius: 12px !important;
        animation: pulseSidebarToggle 2.5s infinite !important;
        transition: all 0.2s ease-in-out !important;
        z-index: 999999 !important; /* Ensure the toggle is never hidden behind other containers */
        box-shadow: {sidebar_toggle_shadow} !important;
        top: 1.2rem !important;
        left: -18px !important;
    }}
    
    /* Animation for the SVG icon itself */
    @keyframes pulseIconScale {{
        0%, 100% {{ transform: scale(1); }}
        50% {{ transform: scale(1.2); }}
    }}

    /* Colorize the native SVG icon to match theme */
    [data-testid="collapsedControl"] svg {{
        color: {sidebar_toggle_icon} !important;
        fill: {sidebar_toggle_icon} !important;
        animation: pulseIconScale 2s infinite ease-in-out !important;
        transition: transform 0.4s ease-in-out !important;
    }}
    
    [data-testid="collapsedControl"]:hover {{
        background-color: {sidebar_toggle_hover_bg} !important;
        transform: scale(1.05) !important;
    }}
    /* Apply hover scale and rotate to toggle icons */
    [data-testid="collapsedControl"]:hover svg {{
        animation: none !important;
        transform: scale(1.1) rotate(180deg) !important;
    }}

    /* Ensure Native Close Icon inside the Open Sidebar is highlighted by default */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebar"] button[kind="header"] {{
        background-color: {sidebar_toggle_bg} !important;
        border: 2px solid {sidebar_toggle_border} !important;
        border-radius: 12px !important;
        animation: pulseSidebarToggle 2.5s infinite !important;
        transition: all 0.2s ease-in-out !important;
    }}
    
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stSidebar"] button[kind="header"]:hover {{
        background-color: {sidebar_toggle_hover_bg} !important;
        transform: scale(1.05) !important;
    }}

    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebar"] button[kind="header"] svg {{
        color: {sidebar_toggle_icon} !important;
        fill: {sidebar_toggle_icon} !important;
        animation: pulseIconScale 2s infinite ease-in-out !important;
        transition: transform 0.4s ease-in-out !important;
    }}

    [data-testid="stSidebarCollapseButton"]:hover svg,
    [data-testid="stSidebar"] button[kind="header"]:hover svg {{
        animation: none !important;
        transform: scale(1.1) rotate(-90deg) !important;
    }}

    [data-testid="stHeader"] {{
        background: {header_bg_color} !important;
        border-bottom: 1px solid {header_border_color} !important;
        box-shadow: {header_shadow} !important;
    }}
    [data-testid="stHeader"] h1,
    [data-testid="stHeader"] h2,
    [data-testid="stHeader"] h3,
    [data-testid="stHeader"] h4,
    [data-testid="stHeader"] h5,
    [data-testid="stHeader"] h6 {{
        color: {header_text_color} !important;
    }}
    [data-testid="stHeader"] h1 span {{
        color: {header_subtitle_color} !important;
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {title_color};
    }}
    .stMarkdownContainer p, label {{
        color: {text_color};
    }}
    
    /* Custom Metric Container */
    .metric-container {{
        background-color: {container_bg};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid {container_border};
    }}

    /* Smooth Metric Fade-In Animation */
    @keyframes fadeInScale {{
        0% {{ opacity: 0; transform: scale(0.95) translateY(10px); }}
        100% {{ opacity: 1; transform: scale(1) translateY(0); }}
    }}
    .metric-animate {{
        animation: fadeInScale 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}

    .metric-title {{
        font-size: 1.2rem;
        color: {title_color};
        margin-bottom: 10px;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: bold;
        color: {val_color};
    }}

    /* Animate DataFrames and Charts globally */
    [data-testid="stDataFrame"] {{
        animation: fadeInScale 0.6s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}
    
    /* Smooth Pie Chart Render Animation */
    [data-testid="stPlotlyChart"] {{
        animation: fadeInScale 0.8s cubic-bezier(0.25, 0.8, 0.25, 1) forwards;
    }}

    /* Professional Dashboard Main Container */
    [data-testid="block-container"] {{
        background-color: {container_bg};
        padding: 3rem 2rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        margin-bottom: 2rem;
        border: 1px solid {container_border};
    }}
    
    /* Medical Blue Buttons */
    .stButton > button[kind="secondary"], .stDownloadButton > button, [data-testid="stDownloadButton"] button, [data-testid="stFormSubmitButton"] button {{
        background-color: #0056b3 !important;
        color: white !important;
        padding: 12px 24px !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }}
    .stButton > button[kind="secondary"]:hover, .stDownloadButton > button:hover, [data-testid="stDownloadButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover {{
        background-color: #004494 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2) !important;
    }}
    .stButton > button[kind="secondary"]:active, .stDownloadButton > button:active, [data-testid="stDownloadButton"] button:active, [data-testid="stFormSubmitButton"] button:active {{
        background-color: #003366 !important; /* Darker blue for click state */
        transform: scale(0.96) !important; /* Subtle scale-down on click */
    }}

    /* Destructive Red Buttons (Primary) */
    [data-testid="stButton"] > button[kind="primary"] {{
        background-color: #D9534F !important;
        color: white !important;
        padding: 12px 24px !important;
        font-size: 1.05rem !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 4px 12px rgba(217, 83, 79, 0.3) !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:hover {{
        background-color: #C9302C !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(217, 83, 79, 0.4) !important;
    }}
    [data-testid="stButton"] > button[kind="primary"]:active {{
        background-color: #AC2925 !important;
        transform: scale(0.96) !important;
    }}

    /* Custom Stand-out Cancel Button */
    div.element-container:has([id^="cancel-btn-highlight"]) {{
        display: none; /* Hide the anchor container */
    }}
    div.element-container:has([id^="cancel-btn-highlight"]) + div.element-container [data-testid="stButton"] > button {{
        background-color: #334155 !important; /* Dark Slate Background */
        color: #FFC107 !important; /* Vibrant Amber Text */
        border: 1px solid #FFC107 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        transition: all 0.15s ease !important;
    }}
    div.element-container:has([id^="cancel-btn-highlight"]) + div.element-container [data-testid="stButton"] > button:hover {{
        background-color: #475569 !important;
        color: #FFD54F !important;
    }}
    div.element-container:has([id^="cancel-btn-highlight"]) + div.element-container [data-testid="stButton"] > button:active {{
        background-color: #1E293B !important;
        transform: scale(0.96) !important;
    }}

    /* Expander Styling (Session History & Metadata) */
    [data-testid="stExpander"] {{
        background-color: {expander_bg} !important;
        border: 2px solid {expander_border} !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
    }}
    [data-testid="stExpander"] summary p {{
        color: {expander_text} !important;
        font-weight: bold;
    }}
    [data-testid="stExpander"] svg {{
        color: {expander_text} !important;
    }}

    /* Chatbot Animations */
    @keyframes fadeInUp {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    .chat-animate {{
        animation: fadeInUp 0.4s ease-out forwards;
    }}

    /* Bouncing Dots Animation */
    @keyframes bounce {{
        0%, 80%, 100% {{ transform: scale(0); }}
        40% {{ transform: scale(1); }}
    }}
    .bouncing-dots > div {{
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #50C878;
        border-radius: 100%;
        animation: bounce 1.4s infinite ease-in-out both;
        margin-right: 4px;
    }}
    .bouncing-dots .dot1 {{ animation-delay: -0.32s; }}
    .bouncing-dots .dot2 {{ animation-delay: -0.16s; }}
    .bouncing-dots .dot3 {{ animation-delay: 0s; }}

    /* Gemini-style Fading/Pulsing Cursor */
    @keyframes geminiBlink {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.5; transform: scale(0.8); }}
    }}
    .gemini-cursor {{
        display: inline-block;
        width: 12px;
        height: 12px;
        background: linear-gradient(135deg, #38BDF8, #50C878);
        border-radius: 50%;
        margin-left: 6px;
        animation: geminiBlink 0.8s infinite alternate;
        vertical-align: middle;
    }}

    /* File Uploader Customization */
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed {text_color} !important;
        background-color: {container_bg} !important;
        border-radius: 12px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease-in-out !important;
        position: relative !important;
    }}
    
    /* Hover & Drag-over effect for Dropzone */
    [data-testid="stFileUploaderDropzone"]:hover,
    [data-testid="stFileUploadDropzone"]:hover {{
        border-color: #007BFF !important; /* Medical Blue */
        background-color: rgba(0, 123, 255, 0.05) !important; /* Very subtle blue tint */
        z-index: 10 !important;
    }}

    /* Hide single-scan anchor */
    div.element-container:has(#single-scan-uploader-target) {{
        display: none;
    }}

    /* Hide default content on hover to make room for custom text (Single Scan Only) */
    div.element-container:has(#single-scan-uploader-target) + div.element-container [data-testid="stFileUploaderDropzone"]:hover > *,
    div.element-container:has(#single-scan-uploader-target) + div.element-container [data-testid="stFileUploadDropzone"]:hover > * {{
        opacity: 0 !important;
        transition: opacity 0.2s ease-in-out !important;
        pointer-events: none !important;
    }}

    /* Add custom text and icon when hovering/dragging (Single Scan Only) */
    div.element-container:has(#single-scan-uploader-target) + div.element-container [data-testid="stFileUploaderDropzone"]:hover::before,
    div.element-container:has(#single-scan-uploader-target) + div.element-container [data-testid="stFileUploadDropzone"]:hover::before {{
        content: '📥 Release to Upload Scan...';
        position: absolute;
        font-size: 1.3rem;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #007BFF;
        pointer-events: none;
        animation: fadeInScale 0.3s ease-out forwards;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999 !important;
    }}

    /* Integrated 'Browse files' Button */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploadDropzone"] button {{
        background-color: rgba(0, 123, 255, 0.1) !important;
        color: #007BFF !important;
        border: 1px solid #007BFF !important;
        border-radius: 20px !important;
        font-weight: 600 !important;
        padding: 4px 16px !important;
        transition: all 0.2s ease-in-out !important;
        z-index: 100 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button:hover,
    [data-testid="stFileUploadDropzone"] button:hover {{
        background-color: #007BFF !important;
        color: #FFFFFF !important;
        transform: scale(1.05) !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.2) !important;
    }}

    /* Danger Animation for Tumor Detection (High Risk) */
    @keyframes pulseDanger {{
        0% {{ box-shadow: 0 0 0 0 rgba(217, 83, 79, 0.8); }}
        70% {{ box-shadow: 0 0 0 30px rgba(217, 83, 79, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(217, 83, 79, 0); }}
    }}
    .danger-animate {{
        animation: pulseDanger 0.8s infinite;
        border: 2px solid #D9534F !important;
    }}

    /* Warning Animation for Cyst Detection (Low Risk) */
    @keyframes pulseWarning {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 193, 7, 0.7); }}
        70% {{ box-shadow: 0 0 0 15px rgba(255, 193, 7, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(255, 193, 7, 0); }}
    }}
    .warning-animate {{
        animation: pulseWarning 2.0s infinite;
        border: 2px solid #FFC107 !important;
    }}
    
    /* Vibrant Red Animation for Stone Detection (Medium Risk) */
    @keyframes pulseStone {{
        0% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); }}
        70% {{ box-shadow: 0 0 0 20px rgba(255, 0, 0, 0); }} 
        100% {{ box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); }}
    }}
    .stone-animate {{
        animation: pulseStone 1.2s infinite; 
        border: 2px solid #FF0000 !important;
    }}

    /* Static Success for Normal Detection (No Risk Factor) */
    .success-animate {{
        border: 2px solid #50C878 !important;
        box-shadow: 0 0 10px rgba(80, 200, 120, 0.2);
    }}

    /* Professional Tab Bar Styling - Button Style */
    .stTabs {{
        margin-top: 2rem;
    }}
    .stTabs [data-baseweb="tab-list"] {{
        display: flex;
        justify-content: space-between; /* Spread buttons evenly */
        gap: 15px; /* Spacing between buttons */
        border-bottom: 2px solid {container_border} !important; /* Visible crease line */
        padding-bottom: 15px !important;
        margin-bottom: 15px !important;
        box-shadow: 0px 8px 15px -10px {header_shadow} !important; /* Depth shadow for the crease */
        position: sticky !important; /* Keep tabs visible when scrolling */
        top: 2.875rem !important; /* Sit just below Streamlit's native header */
        background-color: {bg_color} !important; /* Hide scrolling content behind the tabs */
        z-index: 999 !important; /* Ensure tabs stay above other elements */
        padding-top: 10px !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
        letter-spacing: 0.5px !important;
        font-weight: 600 !important;
        color: #6C757D !important;
        background-color: {container_bg} !important;
        border: 1px solid {container_border} !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        flex: 1 !important; /* Force all tabs to be exactly the same uniform width */
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.02) !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(0, 123, 255, 0.05) !important;
        color: #007BFF !important;
        border-color: rgba(0, 123, 255, 0.3) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 123, 255, 0.1) !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: #007BFF !important;
        background-color: rgba(0, 123, 255, 0.1) !important;
        border-color: #007BFF !important;
        /* Replicate the blue indicator via box-shadow to prevent sliding glitches */
        box-shadow: inset 0 -4px 0 0 #007BFF, 0 4px 8px rgba(0, 123, 255, 0.15) !important;
    }}
    /* Disable the native sliding tab highlight to fix sidebar toggle resize lag */
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}

    /* Gemini Input Glow Animation */
    @keyframes geminiInputGlow {{
        0% {{ box-shadow: 0 0 0 2px #38BDF8, 0 4px 12px rgba(56, 189, 248, 0.2); border-color: #38BDF8 !important; }}
        50% {{ box-shadow: 0 0 0 2px #50C878, 0 4px 12px rgba(80, 200, 120, 0.2); border-color: #50C878 !important; }}
        100% {{ box-shadow: 0 0 0 2px #38BDF8, 0 4px 12px rgba(56, 189, 248, 0.2); border-color: #38BDF8 !important; }}
    }}

    /* Native Chat Message Styling Fixes for Theme Conflicts */
    [data-testid="stChatMessage"] {{
        background-color: {container_bg} !important;
        border: 1px solid {container_border} !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }}
    [data-testid="stChatMessageAvatar"] {{
        background-color: {info_bg} !important;
        border: 1px solid {container_border} !important;
    }}
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
        color: {text_color} !important;
    }}

    /* Chat Input Active/Focus State */
    [data-testid="stChatInput"] {{
        border-radius: 25px !important;
        background-color: {chat_input_bg} !important;
        border: 2px solid #38BDF8 !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.1) !important;
        transition: all 0.3s ease-in-out !important;
    }}
    [data-testid="stChatInput"]:focus-within {{
        animation: geminiInputGlow 2.5s infinite alternate !important;
        transform: translateY(-1px) !important;
    }}
    [data-testid="stChatInput"] > div {{
        background-color: transparent !important;
        border: none !important;
    }}
    [data-testid="stChatInput"] textarea {{
        font-size: 1.05rem !important;
        color: {chat_input_text_color} !important;
        -webkit-text-fill-color: {chat_input_text_color} !important;
        background-color: transparent !important;
        padding: 4px 16px !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        background: linear-gradient(135deg, #38BDF8, #50C878) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
        opacity: 1 !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }}

    /* Alter Progress Bar Color for Batch Processing */
    [data-testid="stProgress"] > div > div > div > div {{
        background-color: #50C878 !important; /* Soft Emerald Green */
    }}

    [data-testid="stStatusWidget"] {{
        background-color: {container_bg} !important;
        border: 1px solid #007BFF !important;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.2) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stStatusWidget"] * {{
        color: #007BFF !important;
        font-weight: 600 !important;
    }}

    /* Red blinking LIVE widget */
@keyframes liveBlink {{
        0% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: 0.35; transform: scale(1.03); }}
        100% {{ opacity: 1; transform: scale(1); }}
    }}
    .live-indicator {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        letter-spacing: 0.2px;
        color: #FF3B3B !important;
        text-transform: uppercase;
        animation: liveBlink 1s infinite linear;
    }}
    .live-indicator::before {{
        content: '';
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #FF3B3B;
        box-shadow: 0 0 0 0 rgba(255, 59, 59, 0.6);
        animation: liveBlink 1s infinite linear;
    }}

    /* Medical Theme Loading Spinner Customization */
    [data-testid="stSpinner"] > div {{
        border-color: rgba(0, 123, 255, 0.15) !important;
        border-top-color: #007BFF !important;
        border-width: 4px !important;
    }}
    [data-testid="stSpinner"] p {{
        color: #007BFF !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }}


    /* Custom Spinning Loader Animation */
    @keyframes spinLoader {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    .spinning-loader {{
        border: 5px solid rgba(0, 123, 255, 0.15);
        border-top: 5px solid #007BFF;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spinLoader 1s linear infinite;
        margin: 20px auto;
    }}
    .loader-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px 0;
        background-color: {container_bg};
        border: 1px solid {container_border};
        border-radius: 12px;
        margin: 20px 0;
    }}
    .loader-text {{
        color: {title_color} !important;
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 10px;
    }}

    
    /* Mobile Responsiveness Additions */
    @media screen and (max-width: 768px) {{
        [data-testid="block-container"] {{
            padding: 1.5rem 1rem !important;
            margin-top: 1rem !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch;
            justify-content: flex-start !important;
            padding-bottom: 10px !important;
            gap: 10px !important;
        }}
        .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{
            display: none;
        }}
        .stTabs [data-baseweb="tab"] {{
            flex: 0 0 auto !important;
            white-space: nowrap !important;
            padding: 10px 16px !important;
            font-size: 0.9rem !important;
        }}
        h1 {{ font-size: 1.8rem !important; }}
        h2 {{ font-size: 1.5rem !important; }}
        h3 {{ font-size: 1.3rem !important; }}
        .metric-value {{ font-size: 1.5rem !important; }}
        .metric-title {{ font-size: 1rem !important; }}
        [data-testid="stChatMessage"] {{ padding: 1rem !important; }}
        [data-testid="collapsedControl"] {{
            left: 10px !important;
            top: 10px !important;
        }}
    }}
    </style>
""", unsafe_allow_html=True)
# --- Define Helper Functions for Data Processing and Model Inference ---

def style_prediction_row(row):
    """Apply color styling to a pandas DataFrame row based on the prediction label."""
    label = row.get('Predicted Label') or row.get('Detection Label') or ''
    if label == 'Tumor':
        return ['background-color: rgba(217, 83, 79, 0.3); color: #D9534F; font-weight: 600'] * len(row)
    elif label == 'Stone':
        return ['background-color: rgba(255, 0, 0, 0.3); color: #FF0000; font-weight: 600'] * len(row)
    elif label == 'Cyst':
        return ['background-color: rgba(255, 193, 7, 0.3); color: #FFC107; font-weight: 600'] * len(row)
    elif label == 'Normal':
        return ['background-color: rgba(80, 200, 120, 0.3); color: #50C878; font-weight: 600'] * len(row)
    elif label in ['Uncertain', 'Invalid Scan']:
        return ['background-color: rgba(136, 136, 136, 0.3); color: #888888; font-weight: 600'] * len(row)
    else:
        return [''] * len(row)

def _render_chat_history_txt_bytes(chat_history, prediction=None, confidence=None):
    """
    Convert the provided chat history into a formatted TXT byte string for downloading.
    Supports both English and Hindi characters natively using utf-8-sig.
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    content = "Renal Vision - Medical Advisory Session History\n"
    content += f"Generated on: {timestamp}\n"
    content += "="*60 + "\n\n"
    
    if prediction is not None and confidence is not None:
        content += "--- Analyzed Medical Scan Results ---\n"
        content += f"Prediction: {prediction}\n"
        content += f"Confidence: {confidence:.2f}%\n"
        content += "-"*60 + "\n\n"
        
    for msg in chat_history or []:
        role = "Medical Student" if msg.get("role") == "user" else "Nephrology Medical Tutor"
        text = msg.get("content", "")
        content += f"{role}:\n{text}\n"
        content += "-"*60 + "\n\n"
        
    return content.encode('utf-8-sig'), "text/plain", "txt"

def load_dicom_image(file_bytes) -> Image.Image:
    """
    Safely parse and convert a DICOM file format into a standard PIL Image.
    
    Args:
        file_bytes (bytes): The raw byte data of the uploaded DICOM file.
        
    Returns:
        Image.Image: The processed image converted into a PIL object.
        
    Raises:
        ImportError: If the 'pydicom' library is not installed in the environment.
    """
    if pydicom is None:
        raise ImportError("pydicom library is not installed. Unable to process DICOM files.")
    dicom = pydicom.dcmread(io.BytesIO(file_bytes))
    pixel_array = dicom.pixel_array
    # Normalize the DICOM pixel array to an 8-bit (0-255) format so it can be converted into a standard PIL Image.
    pixel_array = pixel_array - np.min(pixel_array)
    pixel_array = (pixel_array / np.max(pixel_array) * 255).astype(np.uint8)
    return Image.fromarray(pixel_array)

@st.cache_data(show_spinner=False, max_entries=50)
def get_cached_gradcam(file_bytes, file_extension):
    temp_filename = f"temp_gradcam_{uuid.uuid4().hex}.png"
    try:
        if file_extension == 'dcm':
            img = load_dicom_image(file_bytes)
        else:
            img = Image.open(io.BytesIO(file_bytes))
        img.convert("RGB").save(temp_filename, "PNG")
        pipeline = PredictionPipeline(filename=temp_filename, model_path=ONNX_MODEL_PATH)
        return pipeline.make_gradcam_overlay_base64()
    except FileNotFoundError as e:
        if "Required for XAI visualizations" in str(e):
            return None
        raise e
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@st.cache_data(show_spinner=False, max_entries=50)
def get_cached_attention(file_bytes, file_extension):
    temp_filename = f"temp_attention_{uuid.uuid4().hex}.png"
    try:
        if file_extension == 'dcm':
            img = load_dicom_image(file_bytes)
        else:
            img = Image.open(io.BytesIO(file_bytes))
        img.convert("RGB").save(temp_filename, "PNG")
        pipeline = PredictionPipeline(filename=temp_filename, model_path=ONNX_MODEL_PATH)
        return pipeline.make_attention_overlay_base64(filename=temp_filename)
    except FileNotFoundError as e:
        if "Required for XAI visualizations" in str(e):
            return None
        raise e
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@st.cache_data(show_spinner=False, max_entries=50)
def get_cached_lime(file_bytes, file_extension):
    temp_filename = f"temp_lime_{uuid.uuid4().hex}.png"
    try:
        if file_extension == 'dcm':
            img = load_dicom_image(file_bytes)
        else:
            img = Image.open(io.BytesIO(file_bytes))
        img.convert("RGB").save(temp_filename, "PNG")
        pipeline = PredictionPipeline(filename=temp_filename, model_path=ONNX_MODEL_PATH)
        return pipeline.make_lime_overlay_base64(filename=temp_filename)
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- Define the Main Streamlit Application Layout: Setup the sidebar, main title, and layout containers ---
st.sidebar.markdown(f"""
    <h2 style='color: #0000FF !important; text-shadow: 0 0 8px rgba(0, 0, 255, 0.6); font-weight: 800; text-align: left !important; font-size: 1.4rem; padding-bottom: 0.5rem; border-bottom: 1px solid {sidebar_border}; margin-bottom: 1.5rem;'>
    🏥 Control Center
    </h2>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="background-color: rgba(80, 200, 120, 0.25); padding: 10px 15px; border-radius: 8px; border-left: 4px solid #50C878; margin-bottom: 20px;">
    <strong style="color: #50C878; font-size: 0.95rem;">🟢 System Status: Online</strong>
</div>

<div style="background-color: {info_bg}; padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border};">
    <strong style="color: {title_color}; font-size: 1rem;">📌 Mandatory Instructions</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>1. Data Acquisition:</strong><br>Upload high-resolution scans (DICOM, JPEG, PNG).<br><br>
        <strong>2. Initialization:</strong><br>AI Assistant unlocks upon successful valid scan analysis.<br><br>
        <strong>3. Validation Layer:</strong><br>System rejects chromatic (RGB) or non-medical images to prevent bias.
    </div>
</div>

<div style="background-color: rgba(139, 92, 246, 0.25); padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border}; border-left: 4px solid #8B5CF6; margin-top: 20px;">
    <strong style="color: #8B5CF6; font-size: 1rem;">🧠 Explainable AI (XAI)</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>Multi-Model Explainability:</strong><br>
        To ensure maximum diagnostic transparency, the pipeline utilizes three state-of-the-art XAI frameworks:<br><br>
        🔥 <strong>Grad-CAM:</strong> Class Activation Heatmaps.<br>
        🟢 <strong>Attention Map:</strong> Saliency & Spatial Focus.<br>
        🍋 <strong>LIME:</strong> Superpixel Perturbation.<br>
    </div>
</div>

<div style="background-color: rgba(245, 158, 11, 0.25); padding: 15px; border-radius: 8px; border: 1px solid {sidebar_border}; border-left: 4px solid #F59E0B; margin-top: 20px; margin-bottom: 20px;">
    <strong style="color: #F59E0B; font-size: 1rem;">🌟 App Overview & Features</strong><br><br>
    <div style="font-size: 0.85rem; color: {sidebar_text}; line-height: 1.6;">
        <strong>📑 Batch Processing:</strong><br>Upload multiple CT scans at once to generate a consolidated, downloadable diagnostic report.<br><br>
        <strong>🤖 Renal AI Assistant:</strong><br>Context-aware nephrology chatbot providing supportive advisory based on the latest scan predictions.<br><br>
        <strong>✨ Key Capabilities:</strong><br>
        • High-accuracy Tumor vs. Normal classification.<br>
        • Automated image quality & CT verification.<br>
        • Bilingual AI support (English & Hindi).
    </div>
</div>
""", unsafe_allow_html=True)

col_title, col_toggle = st.columns([8.5, 1.5])
with col_title:
    st.markdown(f"<h1 style='font-weight: 800; letter-spacing: 1px; line-height: 1.3; text-shadow: {main_title_shadow}; text-align: center;'>🩺 Renal Vision<br><span style='display: block; font-size: 0.6em; font-weight: 400; color: #6C757D; text-align: center; justify-content: center; text-shadow: none;'>AI-Powered Diagnostic Dashboard</span></h1>", unsafe_allow_html=True)
with col_toggle:
    st.write("") # Add empty space to vertically align the theme toggle button with the main title.
    
    def toggle_theme():
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        
    toggle_container = st.empty()
    def render_toggle(disabled=False, key_suffix="initial"):
        with toggle_container:
            if st.session_state.theme == "light":
                st.button("🌜 Dark", on_click=toggle_theme, key=f"theme_toggle_dark_{key_suffix}", disabled=disabled)
            else:
                st.button("🌞 Light", on_click=toggle_theme, key=f"theme_toggle_light_{key_suffix}", disabled=disabled)
                
    render_toggle(disabled=False, key_suffix="initial")

st.markdown("##### *DIAGNOSTIC ADVISORY: This AI-powered tool is designed for preliminary screening and decision support only. The results generated by the VGG16 model must be correlated with clinical findings by a certified Radiologist or Nephrologist. This interface does not provide a final medical diagnosis.*")
st.divider()
st.markdown(
    f"""<h3 style=\"text-align: center !important; margin: 0;\">
        <span class=\"live-indicator\" style=\"margin-right: 14px;\">LIVE</span>
        AI-Powered Classification for Renal Diseases (Normal/Cyst/Stone/Tumor)
    </h3>""",
    unsafe_allow_html=True,
)


tab1, tab2, tab3 = st.tabs(["🔬 SINGLE SCAN ANALYSIS", "📑 BATCH MEDICAL REPORT", "🤖 RENAL AI"])

with tab1:
    st.markdown("<span id='single-scan-uploader-target'></span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Medical Scan (JPEG, PNG, DICOM)",type=['jpg', 'jpeg', 'png', 'dcm'], key=f"single_scan_uploader_{st.session_state.single_uploader_key}")
    
    # If the user manually uploads a file, clear any selected sample
    if uploaded_file is not None:
        st.session_state.selected_sample = None

    # --- Sample Image Feature ---
    with st.expander("🧪 Test with Sample Images", expanded=False):
        st.markdown("Don't have a scan? Try the AI with one of these sample images.")
        
        sample_dir = Path("samples")
        if not sample_dir.exists():
            sample_dir.mkdir(parents=True, exist_ok=True)
            st.info("Created 'samples' directory. Please place some test images (e.g., normal.png, tumor.png) inside it to use this feature.")
        
        sample_files = list(sample_dir.glob("*.*"))
        valid_samples = [f for f in sample_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.dcm']]
        
        if valid_samples:
            sample_cols = st.columns(4)
            for i, sample_path in enumerate(valid_samples[:4]):
                with sample_cols[i]:
                    try:
                        if sample_path.suffix.lower() != '.dcm':
                            st.image(str(sample_path), use_container_width=True, caption=sample_path.name)
                        else:
                            st.markdown(f"**DICOM**: {sample_path.name}")
                    except Exception:
                        st.markdown(f"*{sample_path.name}*")
                    
                    if st.button(f"Analyze {sample_path.name}", key=f"btn_{sample_path.name}", use_container_width=True):
                        st.session_state.selected_sample = sample_path
        else:
            st.info("No valid sample images (JPG, PNG, DCM) found in the 'samples/' directory.")

    if st.session_state.get('selected_sample') is not None:
        sample_path = st.session_state.selected_sample
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                file_bytes = f.read()
            
            class MockFile(io.BytesIO):
                def __init__(self, name, data):
                    super().__init__(data)
                    self.name = name
                    self.size = len(data)
            uploaded_file = MockFile(sample_path.name, file_bytes)
        else:
            st.session_state.selected_sample = None
    # --- End Sample Image Feature ---

    if uploaded_file is not None:
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        # Verify if the uploaded file is new. This prevents redundant model inference on the same file during UI reruns.
        if st.session_state.processed_file_id != current_file_id:
            st.session_state.prediction_active = False
            st.session_state.processed_file_id = current_file_id
            
            temp_filename = f"temp_scan_{uuid.uuid4().hex}.png" # Use a unique temp name to prevent concurrency issues
            file_bytes = uploaded_file.getvalue() # Get file bytes once
            file_extension = uploaded_file.name.split('.')[-1].lower()

            # Define these early so they can be cleared even if skeleton is not rendered
            pipeline_status = st.empty()
            step_validate = st.empty()
            preprocessing_status = st.empty()
            status_container = st.empty()
            skeleton_container = st.empty()

            try:
                # --- Image Loading and Conversion ---
                if file_extension == 'dcm':
                    img = load_dicom_image(file_bytes)
                else:
                    img = Image.open(io.BytesIO(file_bytes))
                
                # Save the PIL image to a temporary file for the pipeline
                img.convert("RGB").save(temp_filename, "PNG")

                # --- Prediction using the unified pipeline ---
                pipeline_status.info("⚙️ Initializing AI Diagnostic Pipeline...")
                
                pipeline = PredictionPipeline(filename=temp_filename, model_path=ONNX_MODEL_PATH)
                
                # --- Perform initial validations *before* rendering skeleton ---
                step_validate.info("⏳ Validating image quality & integrity...")
                pipeline.validate_image_quality()
                step_validate.info("⏳ Verifying renal CT characteristics...")
                pipeline.verify_ct_scan()
                step_validate.info("⏳ Applying data preprocessing steps...")
                pipeline.preprocess_image()
                step_validate.empty()

                with preprocessing_status.expander("⚙️ Data Preprocessing Steps Applied", expanded=True):
                    st.markdown("""
                    **Preprocessing Steps Applied:**
                    1. **Resizing:** Image resized to match model input dimensions (224x224).
                    2. **Normalization:** Pixel values scaled to a range of [0, 1].
                    3. **Color Conversion:** Converted to RGB format.
                    """)

                skeleton_container.markdown("""
                <div class="loader-container">
                    <div class="spinning-loader"></div>
                    <div class="loader-text">Analyzing Renal Scan...</div>
                </div>
                """, unsafe_allow_html=True)

                with status_container.container():
                    # Get prediction, Grad-CAM, and Preprocessing previews
                    step4 = st.empty()
                    step4.markdown("⏳ **Phase 1/3:** Running deep learning prediction...")
                    prediction_result = pipeline.predict_detailed()
                    step4.markdown("✅ **Phase 1/3:** Running deep learning prediction...")
                    
                    step5 = st.empty()
                    step5.markdown("⏳ **Phase 2/3:** Generating Grad-CAM visualization...")
                    file_bytes = uploaded_file.getvalue()
                    gradcam_b64 = get_cached_gradcam(file_bytes, file_extension)
                    if gradcam_b64:
                        step5.markdown("✅ **Phase 2/3:** Generating Grad-CAM visualization...")
                    else:
                        step5.markdown("✅ **Phase 2/3:** Grad-CAM bypassed (Keras model missing)...")
                    
                    step6 = st.empty()
                    step6.markdown("⏳ **Phase 3/3:** Finalizing diagnostic previews...")
                    previews_b64 = pipeline.make_preprocess_previews_base64()
                    step6.markdown("✅ **Phase 3/3:** Finalizing diagnostic previews...")

                status_container.empty()
                pipeline_status.empty()
                skeleton_container.empty()
                preprocessing_status.empty()

                predicted_label = prediction_result["prediction"]
                confidence = prediction_result["confidence"]
                
                # --- Update Session State ---
                st.session_state.latest_label = predicted_label
                st.session_state.latest_confidence = confidence * 100
                
                new_entry = {
                    "File Name": uploaded_file.name,
                    "Predicted Label": predicted_label,
                    "Confidence Score": f"{confidence * 100:.2f}%"
                }
                st.session_state.history.insert(0, new_entry)
                st.session_state.history = st.session_state.history[:5]
                
                if predicted_label == 'Tumor':
                    color_indicator = '#D9534F'
                    animation_class = 'danger-animate'
                elif predicted_label == 'Stone':
                    color_indicator = '#FF0000'
                    animation_class = 'stone-animate'
                elif predicted_label == 'Cyst':
                    color_indicator = '#FFC107'
                    animation_class = 'warning-animate'
                elif predicted_label == 'Normal':
                    color_indicator = '#50C878'
                    animation_class = 'success-animate'
                else:
                    color_indicator = '#888888'
                    animation_class = ''

                st.session_state.prediction_active = True
                st.session_state.persisted_label = predicted_label
                st.session_state.persisted_confidence = confidence
                st.session_state.persisted_probabilities = prediction_result.get("probabilities")
                st.session_state.persisted_heatmap = gradcam_b64
                st.session_state.persisted_attention = None
                st.session_state.persisted_lime = None
                st.session_state.persisted_img = img
                st.session_state.persisted_file_bytes = file_bytes
                st.session_state.persisted_file_ext = file_extension
                st.session_state.persisted_preprocessed_img = previews_b64["resized"]
                st.session_state.persisted_color_indicator = color_indicator
                st.session_state.persisted_animation_class = animation_class
                
                if predicted_label == 'Normal':
                    st.snow()

            except ValueError as ve:
                # Catch specific validation errors from the pipeline
                st.error(str(ve))
                st.session_state.processed_file_id = None
            except Exception as e:
                if pydicom and isinstance(e, pydicom.errors.InvalidDicomError):
                    st.error("Error: The uploaded DICOM file is corrupted or invalid. Please provide a valid DICOM file.")
                elif isinstance(e, FileNotFoundError):
                    st.error(f"System Error: {str(e)}")
                else:
                    st.error(f"An unexpected error occurred: {str(e)}")
                st.session_state.processed_file_id = None
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
    
        def clear_single_scan_results():
            st.session_state.prediction_active = False
            st.session_state.latest_label = None
            st.session_state.latest_confidence = None
            st.session_state.persisted_probabilities = None
            st.session_state.show_clear_scan_confirm = False
            st.session_state.selected_sample = None
            st.session_state.generating_attention = False
            st.session_state.generating_lime = False
            st.session_state.single_uploader_key += 1

        def confirm_clear_scan():
            st.session_state.show_clear_scan_confirm = True
            
        def cancel_clear_scan():
            st.session_state.show_clear_scan_confirm = False

        if st.session_state.show_clear_scan_confirm:
            st.warning("Are you sure you want to clear the scan results?")
            _, sc1, sc2 = st.columns([2, 1, 1])
            with sc1:
                st.button("✔️ Yes, Clear", key="yes_clear_scan", type="primary", on_click=clear_single_scan_results, use_container_width=True)
            with sc2:
                st.markdown("<span id='cancel-btn-highlight-scan'></span>", unsafe_allow_html=True)
                st.button("❌ Cancel", key="cancel_clear_scan", on_click=cancel_clear_scan, use_container_width=True)
        else:
            # Keep a neat, right-aligned Clear Results button right below the uploader
            _, btn_clear_col = st.columns([3, 1])
            with btn_clear_col:
                st.button("🧹 Clear Results", key="clear_results_btn", type="primary", on_click=confirm_clear_scan, use_container_width=True)

        # Render the prediction results and metrics conditionally, relying on the active prediction state.
        if st.session_state.prediction_active:
            with st.expander("✅ Scan analysis completed. Click here to view data preprocessing steps.", expanded=False):
                st.markdown("""
                **Preprocessing Steps Applied:**
                1. **Resizing:** Image resized to match model input dimensions (224x224).
                2. **Normalization:** Pixel values scaled to a range of [0, 1].
                3. **Color Conversion:** Converted to RGB format.
                4. **Feature Extraction:** Processed via VGG16 model layers.
                """)
            st.markdown("---")
            
            col_metric, col_chart = st.columns(2, vertical_alignment="center")
            with col_metric:
                st.markdown(f"""
                <div class="metric-container metric-animate {st.session_state.persisted_animation_class}" style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                    <div class="metric-title">Diagnostic Prediction</div>
                    <div class="metric-value" style="color: {st.session_state.persisted_color_indicator}">{st.session_state.persisted_label}</div>
                    <div style="margin-top: 10px; font-size: 1.1rem; color: {text_color};">Confidence: <strong style="color: {st.session_state.persisted_color_indicator};">{st.session_state.persisted_confidence * 100:.1f}%</strong></div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_chart:
                if st.session_state.persisted_probabilities and px is not None:
                    probs = st.session_state.persisted_probabilities
                    df_probs = pd.DataFrame({
                        "Class": list(probs.keys()),
                        "Probability": list(probs.values())
                    })
                    fig = px.pie(
                        df_probs, 
                        names="Class", 
                        values="Probability", 
                        hole=0.4,
                        color="Class",
                        color_discrete_map={
                            "Normal": "#50C878",
                            "Tumor": "#D9534F",
                        "Stone": "#FF0000",
                            "Cyst": "#FFC107"
                        }
                    )
                    fig.update_layout(
                        margin=dict(t=20, b=20, l=20, r=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color=text_color),
                    legend=dict(font=dict(color="#000000" if st.session_state.theme == 'light' else text_color))
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            if st.session_state.persisted_label == 'Uncertain':
                st.warning("⚠️ **Disclaimer:** The model's confidence is below 60%. The AI cannot definitively identify the exact class or disease in this scan. Clinical correlation by a medical professional is strictly required.")

            st.markdown(f'<h3 style="color: {title_color}; font-weight: bold; text-align: center;">Primary Imaging & Preprocessing</h3>', unsafe_allow_html=True)
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.image(st.session_state.persisted_img, use_container_width=True)
                st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Original Uploaded Scan</strong></p>', unsafe_allow_html=True)
            with img_col2:
                st.image(st.session_state.persisted_preprocessed_img, use_container_width=True)
                st.markdown(f'<p style="text-align: center; color: {text_color}; font-size: 1.1rem;"><strong>Preprocessed (224x224)</strong></p>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(f'<h3 style="color: {title_color}; font-weight: bold; text-align: center;">Advanced Explainable AI (XAI) Analysis</h3>', unsafe_allow_html=True)
            
            # Decode XAI images from data URLs back to bytes for downloading
            gradcam_bytes = base64.b64decode(st.session_state.persisted_heatmap.split(',')[1]) if st.session_state.persisted_heatmap else b""

            xai_tab1, xai_tab2 = st.tabs(["🔥 Grad-CAM", "🟢 Attention Map"])
            
            with xai_tab1:
                x_col1, x_col2 = st.columns([1, 1])
                with x_col1:
                    if st.session_state.persisted_heatmap:
                        st.image(st.session_state.persisted_heatmap, use_container_width=True)
                        st.download_button(
                            label="📥 Download Grad-CAM",
                            data=gradcam_bytes,
                            file_name="gradcam_overlay.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ Grad-CAM visualization is unavailable because the required Keras (.h5) model was not found. Please explore the LIME tab for ONNX-compatible explainability.")
                with x_col2:
                    st.markdown(f"<h4 style='color: {title_color};'>Grad-CAM (Gradient-weighted Class Activation Mapping)</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: {text_color};'>Highlights the most important regions driving the prediction by using the gradients of the target concept flowing into the final convolutional layer.</p>", unsafe_allow_html=True)
                    if st.session_state.persisted_label == 'Tumor':
                        st.info("🔬 **Analysis Note:** The model focused heavily on the lower cortical region due to abnormal density, suggesting a potential solid mass.")
                    elif st.session_state.persisted_label == 'Stone':
                        st.info("🔬 **Analysis Note:** The model focused on high-density hyperattenuating structures typical of renal calculi (stones).")
                    elif st.session_state.persisted_label == 'Cyst':
                        st.info("🔬 **Analysis Note:** The model highlighted a well-circumscribed, fluid-filled region indicative of a cyst.")
                    elif st.session_state.persisted_label == 'Uncertain':
                        st.warning("🔬 **Analysis Note:** Model activations are ambiguous. The heatmaps may highlight diffuse or irrelevant regions due to low confidence.")
                    else:
                        st.success("🔬 **Analysis Note:** The model verified uniform parenchymal density without anomalous focal lesions, distributing focus appropriately.")

            with xai_tab2:
                x_col1, x_col2 = st.columns([1, 1])
                with x_col1:
                    @st.fragment
                    def render_attention_tab():
                        gen_container = st.empty()
                        if st.session_state.persisted_attention is None:
                            with gen_container.container():
                                st.info("Click below to generate the Attention Map.")
                                if st.button("🟢 Generate Attention Map", key="btn_gen_attention", use_container_width=True):
                                    with st.spinner("⏳ Generating Attention Map..."):
                                        try:
                                            res = get_cached_attention(st.session_state.persisted_file_bytes, st.session_state.persisted_file_ext)
                                            if res is None:
                                                st.warning("⚠️ Attention Map requires the original Keras (.h5) model, which is missing. Try using LIME instead.")
                                            else:
                                                st.session_state.persisted_attention = res
                                        except Exception as e:
                                            st.error(f"Error generating Attention Map: {e}")
                                            
                        if st.session_state.persisted_attention:
                            gen_container.empty()
                            attention_bytes = base64.b64decode(st.session_state.persisted_attention.split(',')[1])
                            st.image(st.session_state.persisted_attention, use_container_width=True)
                            st.download_button(
                                label="📥 Download Attention Map",
                                data=attention_bytes,
                                file_name="attention_overlay.png",
                                mime="image/png",
                                use_container_width=True
                            )
                    render_attention_tab()
                with x_col2:
                    st.markdown(f"<h4 style='color: {title_color};'>Attention Map (Saliency)</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='color: {text_color};'>Explains the prediction by mapping the gradients of the output to the input image, highlighting the exact pixels (fine-grained attention) the model focused on.</p>", unsafe_allow_html=True)
                    if st.session_state.persisted_label == 'Tumor':
                        st.info("🔬 **Analysis Note:** High-intensity attention pixels correspond to irregular margins along the renal pelvis and cortex contour.")
                    elif st.session_state.persisted_label == 'Stone':
                        st.info("🔬 **Analysis Note:** High-intensity attention pixels correspond to the calcified mass within the renal collecting system.")
                    elif st.session_state.persisted_label == 'Cyst':
                        st.info("🔬 **Analysis Note:** High-intensity attention pixels define the smooth boundaries of the cortical cystic lesion.")
                    elif st.session_state.persisted_label == 'Uncertain':
                        st.warning("🔬 **Analysis Note:** Attention pixels are scattered with low overall importance, reflecting the model's inability to confidently classify the image.")
                    else:
                        st.success("🔬 **Analysis Note:** The attention pixels evenly cover the renal cortex, confirming normal structural integrity without focal anomalies.")

            # with xai_tab3:
            #     x_col1, x_col2 = st.columns([1, 1])
            #     with x_col1:
            #         @st.fragment
            #         def render_lime_tab():
            #             gen_container = st.empty()
            #             if st.session_state.persisted_lime is None:
            #                 with gen_container.container():
            #                     st.info("Click below to generate the LIME Analysis.")
            #                     if st.button("🍋 Generate LIME Analysis", key="btn_gen_lime", use_container_width=True):
            #                         with st.spinner("⏳ Generating LIME Analysis..."):
            #                             try:
            #                                 res = get_cached_lime(st.session_state.persisted_file_bytes, st.session_state.persisted_file_ext)
            #                                 st.session_state.persisted_lime = res
            #                             except Exception as e:
            #                                 st.error(f"Error generating LIME Analysis: {e}")
                                            
            #             if st.session_state.persisted_lime:
            #                 gen_container.empty()
            #                 lime_bytes = base64.b64decode(st.session_state.persisted_lime.split(',')[1])
            #                 st.image(st.session_state.persisted_lime, use_container_width=True)
            #                 st.download_button(
            #                     label="📥 Download LIME",
            #                     data=lime_bytes,
            #                     file_name="lime_overlay.png",
            #                     mime="image/png",
            #                     use_container_width=True
            #                 )
            #         render_lime_tab()
            #     with x_col2:
            #         st.markdown(f"<h4 style='color: {title_color};'>LIME (Local Interpretable Model-agnostic Explanations)</h4>", unsafe_allow_html=True)
            #         st.markdown(f"<p style='color: {text_color};'>Explains the prediction by perturbing the input image (breaking it into superpixels) and observing how the predictions change, identifying the exact superpixels that contributed to the class.</p>", unsafe_allow_html=True)
            #         if st.session_state.persisted_label == 'Tumor':
            #             st.info("🔬 **Analysis Note:** Positive superpixels correspond to irregular margins along the renal pelvis and cortex contour.")
            #         elif st.session_state.persisted_label == 'Stone':
            #             st.info("🔬 **Analysis Note:** Positive superpixels correspond to the calcified mass within the renal collecting system.")
            #         elif st.session_state.persisted_label == 'Cyst':
            #             st.info("🔬 **Analysis Note:** Positive superpixels define the smooth boundaries of the cortical cystic lesion.")
            #         elif st.session_state.persisted_label == 'Uncertain':
            #             st.warning("🔬 **Analysis Note:** Superpixels are scattered with low overall importance, reflecting the model's inability to confidently classify the image.")
            #         else:
            #             st.success("🔬 **Analysis Note:** The superpixels evenly cover the renal cortex, confirming normal structural integrity.")
                
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        # Reset the session state variables when the user clears or removes the uploaded file to prepare for a new session.
        st.session_state.processed_file_id = None
        st.session_state.prediction_active = False
        st.session_state.latest_label = None
        st.session_state.latest_confidence = None
        st.session_state.persisted_preprocessed_img = None
        st.session_state.single_error = None
        st.session_state.single_result_temp = None
        st.session_state.selected_sample = None
        st.session_state.generating_attention = False
        st.session_state.generating_lime = False

    # --- Render Session History & Metadata Section: Display a table of past predictions within an expander ---
    st.markdown("---")
    with st.expander("📋 Session History "):
        if st.session_state.history:
            history_df = pd.DataFrame(st.session_state.history)
            history_df.index = history_df.index + 1
            styled_history_df = history_df.style.apply(style_prediction_row, axis=1)
            st.dataframe(styled_history_df, use_container_width=True)
        else:
            st.info("No session records available yet.")

with tab2:
    st.markdown(f"""
    <div class="metric-animate" style="background-color: {info_bg}; padding: 15px; border-radius: 10px; border: 1px solid {title_color}; margin-bottom: 20px;">
        <strong style="color: {title_color}; font-size: 1.1rem;">BATCH PROCESSING PROTOCOL:</strong><br><br>
        <span style="color: {text_color};">
        You may upload multiple CT scans simultaneously for high-throughput screening. The system will process each scan sequentially and generate a consolidated diagnostic summary table.
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    batch_files = st.file_uploader("Drop multiple renal scans here (JPEG, PNG, DICOM)", type=['jpg', 'jpeg', 'png', 'dcm'], accept_multiple_files=True, key=f"batch_scan_uploader_{st.session_state.batch_uploader_key}")
    
    # Compare the current batch of files with the previous one. Clear cached results if a new batch is detected to avoid mixing data.
    current_batch_ids = [f"{b.name}_{b.size}" for b in batch_files] if batch_files else []
    if st.session_state.batch_file_ids != current_batch_ids:
        st.session_state.batch_results = None
        st.session_state.batch_pdf_bytes = None
        st.session_state.batch_csv_bytes = None
        st.session_state.batch_file_ids = current_batch_ids
    
    def background_batch_process(files_data, session_state):
        try:
            batch_results = []
            temp_filename = f"temp_batch_scan_{uuid.uuid4().hex}.png"
            for i, file_data in enumerate(files_data):
                session_state.batch_status_message = f"Processing {file_data['name']} ({i+1}/{len(files_data)})..."
                
                # Generate an MD5 hash of the file bytes to check if we've seen this exact scan before
                file_hash = hashlib.md5(file_data['bytes']).hexdigest()
                
                if file_hash in session_state.batch_prediction_cache:
                    cached_res = session_state.batch_prediction_cache[file_hash].copy()
                    cached_res["File Name"] = file_data['name'] # Keep accurate filename if duplicate has a different name
                    batch_results.append(cached_res)
                    
                    # Feed the latest prediction directly into global context for the Chatbot
                    session_state.latest_label = cached_res["Detection Label"]
                    session_state.latest_confidence = float(cached_res["Confidence (%)"])
                    
                    session_state.batch_progress = (i + 1) / len(files_data)
                    continue

                try:
                    if file_data['ext'] == 'dcm':
                        img = load_dicom_image(file_data['bytes'])
                    else:
                        img = Image.open(io.BytesIO(file_data['bytes']))
                    
                    img.convert("RGB").save(temp_filename, "PNG")
                    pipeline = PredictionPipeline(filename=temp_filename, model_path=ONNX_MODEL_PATH)
                    pipeline.validate_image_quality()
                    
                    prediction_result = pipeline.predict_detailed()
                    res = {
                        "File Name": file_data['name'],
                        "Detection Label": "Invalid Scan" if prediction_result["confidence"] == 0 else prediction_result["prediction"],
                        "Confidence (%)": f"{prediction_result['confidence'] * 100:.2f}"
                    }
                    batch_results.append(res)
                    session_state.batch_prediction_cache[file_hash] = res
                    
                    # Feed the latest prediction directly into global context for the Chatbot
                    session_state.latest_label = prediction_result["prediction"]
                    session_state.latest_confidence = prediction_result["confidence"] * 100
                except Exception as e:
                    batch_results.append({
                        "File Name": file_data['name'],
                        "Detection Label": "Invalid Scan",
                        "Confidence (%)": "0.00"
                    })
                session_state.batch_progress = (i + 1) / len(files_data)
                session_state.batch_results_temp = batch_results.copy()
                
            session_state.batch_results_temp = batch_results
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
            session_state.batch_processing_running = False
            session_state.batch_status_message = "✅ Batch processing complete."

    if batch_files:
        def clear_batch_results():
            st.session_state.batch_results = None
            st.session_state.batch_pdf_bytes = None
            st.session_state.batch_csv_bytes = None
            st.session_state.batch_file_ids = []
            st.session_state.batch_processing_running = False
            st.session_state.batch_progress = 0.0
            st.session_state.batch_results_temp = None
            st.session_state.batch_status_message = ""
            st.session_state.show_clear_batch_confirm = False
            st.session_state.batch_uploader_key += 1

        def confirm_clear_batch():
            st.session_state.show_clear_batch_confirm = True
            
        def cancel_clear_batch():
            st.session_state.show_clear_batch_confirm = False

        if st.session_state.show_clear_batch_confirm:
            st.warning("Are you sure you want to clear the batch results?")
            sc1, sc2, _ = st.columns([1.5, 1.5, 2])
            with sc1:
                st.button("✔️ Yes, Clear", key="yes_clear_batch", type="primary", on_click=clear_batch_results, use_container_width=True)
            with sc2:
                st.markdown("<span id='cancel-btn-highlight-batch'></span>", unsafe_allow_html=True)
                st.button("❌ Cancel", key="cancel_clear_batch", on_click=cancel_clear_batch, use_container_width=True)
        else:
            col1, col2, col3 = st.columns([1.5, 1.5, 2])
            with col1:
                if st.button("Start Batch Processing", disabled=st.session_state.batch_processing_running, use_container_width=True):
                    st.session_state.batch_processing_running = True
                    st.session_state.batch_progress = 0.0
                    st.session_state.batch_results_temp = None
                    
                    files_data = [
                        {'name': b.name, 'bytes': b.getvalue(), 'ext': b.name.split('.')[-1].lower()} 
                        for b in batch_files
                    ]
                    t = threading.Thread(target=background_batch_process, args=(files_data, st.session_state))
                    add_script_run_ctx(t)
                    t.start()
            with col2:
                if st.session_state.batch_results is not None or st.session_state.batch_processing_running:
                    st.button("🧹 Clear Results", key="clear_batch_btn", type="primary", on_click=confirm_clear_batch, use_container_width=True)

        @st.fragment(run_every="1s")
        def render_batch_status():
            if st.session_state.batch_processing_running:
                st.markdown(f"<span style='color: #007BFF; font-weight: 500;'>⏳ {st.session_state.batch_status_message}</span>", unsafe_allow_html=True)
                st.progress(st.session_state.batch_progress)
                if st.session_state.batch_results_temp is not None and len(st.session_state.batch_results_temp) > 0:
                    temp_df = pd.DataFrame(st.session_state.batch_results_temp)
                    styled_temp_df = temp_df.style.apply(style_prediction_row, axis=1)
                    st.dataframe(styled_temp_df, use_container_width=True)
            elif st.session_state.batch_results_temp is not None:
                st.markdown(f"<span style='color: #50C878; font-weight: bold;'>{st.session_state.batch_status_message}</span>", unsafe_allow_html=True)
                results_df = pd.DataFrame(st.session_state.batch_results_temp)
                st.session_state.batch_results = results_df
                
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.pdfgen import canvas
                    from reportlab.lib import colors
                    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
                    from reportlab.lib.styles import getSampleStyleSheet
                    
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    elements = []
                    styles = getSampleStyleSheet()
                    
                    elements.append(Paragraph("Batch Medical Report", styles['Title']))
                    elements.append(Spacer(1, 12))
                    
                    data = [["File Name", "Detection Label", "Confidence (%)"]]
                    for res in st.session_state.batch_results_temp:
                        data.append([res["File Name"], res["Detection Label"], res["Confidence (%)"]])
                        
                    t = Table(data)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.grey),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0,0), (-1,0), 12),
                        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                        ('GRID', (0,0), (-1,-1), 1, colors.black)
                    ]))
                    elements.append(t)
                    doc.build(elements)

                    st.session_state.batch_pdf_bytes = buffer.getvalue()
                    st.session_state.batch_csv_bytes = None
                    
                except ImportError:
                    st.session_state.batch_csv_bytes = results_df.to_csv(index=False).encode('utf-8')
                    st.session_state.batch_pdf_bytes = None

                st.session_state.batch_results_temp = None
                
                if hasattr(st, "rerun"):
                    st.rerun()
                else:
                    st.experimental_rerun()
                    
        render_batch_status()

        # Ensure the batch results dataframe and download buttons remain visible across Streamlit reruns
        if st.session_state.batch_results is not None:
            styled_batch_df = st.session_state.batch_results.style.apply(style_prediction_row, axis=1)
            st.dataframe(styled_batch_df, use_container_width=True)
            
            if st.session_state.batch_pdf_bytes is not None:
                st.download_button(
                    label="Download Batch Report as PDF",
                    data=st.session_state.batch_pdf_bytes,
                    file_name="batch_medical_report.pdf",
                    mime="application/pdf"
                )
            elif st.session_state.batch_csv_bytes is not None:
                st.warning("To generate PDF reports, please install 'reportlab' (pip install reportlab). For now, you can download the results as CSV.")
                st.download_button(
                    label="Download Batch Report as CSV",
                    data=st.session_state.batch_csv_bytes,
                    file_name="batch_medical_report.csv",
                    mime="text/csv"
                )

# --- Configure the Medical Advisory Chatbot Assistant: Set up API keys and environment variables required by the Groq SDK ---
# Attempt to load API key securely from Streamlit Secrets (for Streamlit Cloud and local secrets.toml)
if "GROQ_API_KEY" in st.secrets:
    CHATBOT_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # Fallback to local environment variable if secrets are not configured
    CHATBOT_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")

if CHATBOT_API_KEY != "YOUR_API_KEY_HERE":
    CHATBOT_API_KEY = CHATBOT_API_KEY.strip()
    os.environ["GROQ_API_KEY"] = CHATBOT_API_KEY

CHATBOT_MODEL_NAME = "llama-3.3-70b-versatile"

# --- Define the Integrated Medical Advisory Chatbot UI and Logic: Setup the tab interface, warning messages, and chat interactions ---
avatar_grad_start = "#38BDF8"
avatar_grad_end = "#50C878"

with tab3:
    st.markdown("<h5 style='text-align: center; color: #6C757D;'>Active Clinical Session: Context-Aware Nephrology Assistant</h5>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Language Support Instruction Box
    st.markdown(f"""
    <div class="metric-animate" style="background-color: {info_bg}; padding: 15px 20px; border-radius: 8px; border-left: 5px solid {title_color}; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
        <strong style="color: {title_color}; font-size: 1.05rem;">🗣️ Supported Languages / भाषा समर्थन</strong><br>
        <span style="color: {text_color}; font-size: 0.95rem;">
        This Medical Assistant fluently supports both <strong>English</strong> and <strong>Hindi</strong>. Feel free to ask your questions in your preferred language.<br>
        यह मेडिकल असिस्टेंट <strong>अंग्रेजी</strong> और <strong>हिंदी</strong> दोनों भाषाओं का समर्थन करता है। बेझिझक अपनी पसंदीदा भाषा में प्रश्न पूछें।
        </span>
    </div>
    """, unsafe_allow_html=True)

    latest_label = st.session_state.latest_label

    if latest_label == 'Tumor':
        st.markdown(f"""
        <div class="metric-animate danger-animate" style="background-color: {warning_bg}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong style="color: #D9534F; font-size: 1.1rem;">⚠️ Clinical Advisory (Tumor Detected):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Consult a Nephrologist or Oncologist</strong> immediately for a formal diagnosis.<br>
            2. <strong>Recommended further tests</strong>: Consider ordering a biopsy, contrast-enhanced MRI, or PET scan.<br>
            3. Please ensure all medical imaging is formally reviewed by a certified radiologist.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif latest_label == 'Stone':
        st.markdown(f"""
            <div class="metric-animate stone-animate" style="background-color: {warning_bg}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <strong style="color: #FF0000; font-size: 1.1rem;">⚠️ Clinical Advisory (Stone Detected):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Consult a Urologist</strong> for a formal evaluation and treatment plan.<br>
            2. <strong>Recommended actions</strong>: Increase fluid intake and consider further imaging (like non-contrast CT) to determine calculus size and exact location.<br>
            3. Please ensure all medical imaging is formally reviewed by a certified professional.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif latest_label == 'Cyst':
        st.markdown(f"""
    <div class="metric-animate warning-animate" style="background-color: rgba(255, 193, 7, 0.3); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong style="color: #FFC107; font-size: 1.1rem;">⚠️ Clinical Advisory (Cyst Detected):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Consult a Physician</strong> to classify the cyst (simple vs. complex) using the Bosniak classification system.<br>
            2. <strong>Recommended actions</strong>: Most simple cysts are benign, but complex cysts may require routine follow-up or MRI.<br>
            3. Please ensure all medical imaging is formally reviewed by a certified professional.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif latest_label == 'Normal':
        st.markdown(f"""
        <div class="metric-animate success-animate" style="background-color: {success_bg}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong style="color: #50C878; font-size: 1.1rem;">✅ Clinical Advisory (Normal Scan):</strong><br><br>
            <span style="color: {text_color};">
            1. <strong>Wellness Instructions</strong>: Maintain a healthy diet, stay hydrated, and monitor kidney function annually if the patient is high risk.<br>
            2. Routine follow-up is recommended as per the primary care physician's schedule.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif latest_label == 'Uncertain':
        st.markdown(f"""
    <div class="metric-animate danger-animate" style="background-color: rgba(217, 83, 79, 0.3); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <strong style="color: #D9534F; font-size: 1.1rem;">🚨 ACTION REQUIRED: Manual Human Review Needed</strong><br><br>
            <span style="color: {text_color};">
            The model's confidence is below 60% and cannot definitively classify the scan.<br>
            1. <strong>Do not rely on this prediction</strong>. This scan must be forwarded to a specialist for manual evaluation.<br>
            2. Consider repeating the scan if there are imaging artifacts or poor contrast.<br>
            3. Please ensure all medical imaging is formally reviewed by a certified professional.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="metric-animate" style="background-color: {info_bg}; padding: 15px; border-radius: 10px; border: 1px solid {title_color}; margin-bottom: 20px;">
                <strong style="color: {title_color}; font-size: 1.1rem;">🔒 System Status: Chatbot Locked</strong><br><br>
            <span style="color: {text_color};">
                No renal scan has been uploaded yet. Please provide the image first in the Analysis tabs to unlock personalized context-aware advisory.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<h5 style='color: #6C757D; margin-top: 15px;'>Feel free to ask your doubts to Renal Vision AI.</h5>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888; font-size: 0.85rem; margin-top: 5px;'><em>Renal AI can make mistakes, please double-check it.</em></p>", unsafe_allow_html=True)

    st.markdown("<span id='chat-scroll-target'></span>", unsafe_allow_html=True)
    # Assign a fixed height only if there are messages, allowing the input box to move up when cleared
    if len(st.session_state.chat_history) > 0:
        chat_container = st.container(height=550, border=False)
    else:
        chat_container = st.container(border=False)
        
    with chat_container:
        if not st.session_state.chat_history:
            with st.chat_message("assistant", avatar="🩺"):
                welcome_message = "Welcome! I am the Renal AI Assistant. After you upload a scan, I can provide detailed context. Feel free to ask me any general questions about kidney health."
                st.markdown(welcome_message)
        else:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message['content'])
                else:
                    with st.chat_message("assistant", avatar="🩺"):
                        st.markdown(message['content'])
        active_chat_placeholder = st.container()
        
        # Inject an anchor and Javascript to automatically scroll to the bottom of the chat container
        st.markdown("<div id='chat-end'></div>", unsafe_allow_html=True)
        st.components.v1.html(
            """
            <script>
                const doc = window.parent.document;
                const chatEnd = doc.getElementById('chat-end');
                if (chatEnd) {
                    let el = chatEnd;
                    let scrollContainer = null;
                    while (el && el !== doc.body) {
                        const style = doc.defaultView.getComputedStyle(el);
                        if (style.overflowY === 'auto' || style.overflowY === 'scroll' || style.overflow === 'auto' || style.overflow === 'scroll') {
                            scrollContainer = el;
                            break;
                        }
                        el = el.parentElement;
                    }
                    
                    if (scrollContainer) {
                        // Initial smooth scroll to bottom when a new message is sent
                        scrollContainer.scrollTo({ top: scrollContainer.scrollHeight, behavior: 'smooth' });
                        
                        // Continuous instant scroll while the AI is actively streaming text
                        if (!scrollContainer.dataset.autoScrollObserver) {
                            scrollContainer.dataset.autoScrollObserver = 'true';
                            const observer = new MutationObserver(() => {
                                const isNearBottom = scrollContainer.scrollHeight - scrollContainer.clientHeight - scrollContainer.scrollTop < 150;
                                if (isNearBottom) {
                                    scrollContainer.scrollTo({ top: scrollContainer.scrollHeight });
                                }
                            });
                            observer.observe(scrollContainer, { childList: true, subtree: true, characterData: true });
                        }
                    }
                }
            </script>
            """,
            height=0
        )

    prompt = st.chat_input("Type your message here and press Enter...")

    if prompt:
        render_toggle(disabled=True, key_suffix="generating")
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with active_chat_placeholder:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            with st.chat_message("assistant", avatar="✨"):
                assistant_placeholder = st.empty()
                assistant_placeholder.markdown("<span class='is-assistant'></span><div class='bouncing-dots'><div class='dot1'></div><div class='dot2'></div><div class='dot3'></div></div>", unsafe_allow_html=True)

                # Handle the AI API integration: Verify if a scan has been processed first, then call the Groq LLM API if a valid key is provided.
                if not latest_label:
                    response = "⚠️ Please Provide the Image first. The Medical Advisory Chatbot remains locked until a valid scan is analyzed."
                    assistant_placeholder.error(response)
                elif CHATBOT_API_KEY != "YOUR_API_KEY_HERE":
                    try:
                        # Initialize the Groq client. The SDK automatically utilizes the 'GROQ_API_KEY' environment variable configured earlier.
                        client = Groq()
                        
                        system_prompt =  f"""
Current Case Context:
- Diagnostic Prediction: {st.session_state.latest_label}
- Model Confidence: {st.session_state.latest_confidence:.2f}%

# APP CAPABILITIES & CONTEXT
You are integrated into the "Renal Vision" application. Be aware of its capabilities so you can reference them if users ask:
*   **Explainable AI (XAI):** 
    *   **Grad-CAM:** Uses gradients of the target concept flowing into the final convolutional layer to produce a localization map highlighting important regions.
    *   **Attention Map:** Explains the prediction by mapping the gradients of the output to the input image, highlighting the exact pixels the model focused on.
    *   **LIME:** Explains the prediction by perturbing the input image (breaking it into superpixels) and observing how the predictions change.
*   **Batch Processing:** Can process multiple CT scans simultaneously to generate a consolidated diagnostic PDF/CSV report.
*   **Safety & Validation:** Automatically validates image quality and verifies if the uploaded image is a valid renal CT scan before processing.

# UI/OUTPUT FORMATTING (IMPORTANT)
- Use clean, professional markdown with consistent font sizing.
- Prefer: short headings (###), concise bullets (-), and short paragraphs.
- DO NOT include huge ASCII art, giant emojis, or very large markdown ,use small subtitles instead.
- Keep text compact: each section 1-4 bullets.

# ROLE & PERSONALITY
You are an expert Nephrology Medical Assistant, possessing deep, specialized knowledge of all aspects of kidney health. Your expertise includes, but is not limited to:
*   **Renal Masses & Growths:** Benign and malignant tumors, complex and simple cysts (Bosniak classification), and oncocytomas.
*   **Calculi:** Kidney stones (calcium oxalate, uric acid, struvite, cystine), their formation, prevention, and management.
*   **Chronic & Acute Conditions:** Chronic Kidney Disease (CKD) stages, Acute Kidney Injury (AKI), glomerulonephritis, and polycystic kidney disease (PKD).
*   **Systemic Impacts:** Renal hypertension, diabetic nephropathy, and fluid/electrolyte imbalances.

Your tone should be engaging, friendly, empathetic, highly informative, and accessible. Use approachable language with light emojis. Avoid overly dense medical jargon where simple explanations work better, but never sacrifice clinical accuracy. Always explain the **cause behind conditions or predictions**.

# RESPONSE GUIDELINES
*   **Structure:** Use clear headings, bullet points, and bold text to make complex medical data intuitive and scannable.
*   **Clarity:** Explain *why* a condition happens, not just *what* it is. Always include cause-and-effect reasoning. Use analogies if they help make renal physiology easier for a patient to understand.
*   **Precision:** When discussing lab values (like eGFR, Creatinine, or BUN) or imaging results, be exact and detailed.
*   **Style:** Minimal, precise, on-point answers, but with friendly flow.

# CONTEXT-SPECIFIC INSTRUCTIONS
*   **Tumor:** Adopt an empathetic but urgent tone. Explain causes (e.g., genetic mutations, chronic irritation, family history). Discuss potential next steps like contrast-enhanced MRI, biopsy, and oncology consultations.
*   **Stone:** Explain why stones form (e.g., concentrated urine, dehydration, high oxalate intake, metabolic disorders). Focus on urological management, hydration strategies, dietary changes, and possible interventions (e.g., lithotripsy).
*   **Cyst:** Explain the Bosniak classification system. Describe causes (e.g., blocked tubules, genetic predisposition, age-related changes). Reassure that simple cysts are often benign, and discuss follow-up imaging.
*   **Normal:** Provide general kidney health and wellness tips. Explain why prevention matters (hydration, balanced diet, blood pressure control).
*   **Uncertain:** Strongly emphasize the need for a manual review by a radiologist. Explain possible causes of uncertainty (image quality, overlapping features, atypical presentation). Avoid giving definitive diagnostic advice.

# LANGUAGE & TONE
- Use simple, everyday English.
- Be clear, concise, and well-structured.
- Warm, friendly, and encouraging tone with appropriate emojis.
- Explain medical terms and related keywords in easy language.
- State that AI predictions are informational and should be reviewed by a healthcare professional.

# MANDATORY SAFETY CONSTRAINT
Every single response you generate MUST conclude with a specific safety disclaimer, separated from the main body of text by a horizontal rule (`---`). 

1.  **For English Conversations:** You must include this exact text at the very end of your response:
    > "Renal AI can make mistakes, double check it."

2.  **For Hindi Conversations (including Hinglish/Hindi in Latin script):** You must translate the constraint and include this exact text at the very end of your response:
    > "रीनल एआई (Renal AI) गलतियाँ कर सकता है, कृपया इसे दोबारा जांच लें।"

Never omit this disclaimer, regardless of how simple or complex the user's query is.
"""

                        messages = [
                            {"role": "system", "content": system_prompt}
                        ]
                        for msg in st.session_state.chat_history:
                            messages.append({"role": msg["role"], "content": msg["content"]})
                            
                        completion = client.chat.completions.create(
                            model=CHATBOT_MODEL_NAME,
                            messages=messages,
                            temperature=1,
                            max_completion_tokens=1024,
                            top_p=1,
                            stream=True,
                            stop=None
                        )
                        response = ""

                        for chunk in completion:
                            chunk_text = chunk.choices[0].delta.content or ""
                            if chunk_text:
                                response += chunk_text
                                assistant_placeholder.markdown(f"<span class='is-assistant'></span>{response}▌", unsafe_allow_html=True)
                                time.sleep(0.015) # Enforce a smooth 60fps-like stream by decoupling from fast batched updates
                                
                        # Final render without the cursor to lock the completed message cleanly
                        # Normalize markdown rendering: keep text compact and avoid oversized headers
                        response = response.replace("\n\n\n", "\n\n").strip()
                        # Clamp any accidental very-large markdown headers the model might output
                        response = response.replace("# ", "").replace("## ", "### ")
                    except Exception as e:
                        response = f"Error calling Groq API: {e}"
                else:
                    response = "I have provided the standard medical advisory above. Please configure your Groq API key securely in Streamlit Secrets to enable advanced interactive chat capabilities."

        # Post-check: enforce mandatory safety disclaimer (prevents prompt jailbreaks)
        disclaimer_en = "Renal AI can make mistakes, double check it."
        disclaimer_hi = "रीनल एआई (Renal AI) गलतियाँ कर सकता है, कृपया इसे दोबारा जांच लें।"
        if "---" in response:
            body, tail = response.split("---", 1)
        else:
            body, tail = response, ""

        # Detect language heuristically
        contains_hindi = any(ch in response for ch in ["अ","आ","इ","ई","उ","ऊ","ए","ऐ","ओ","औ","क","ख","ग","घ","च","छ","ज","झ","ट","ठ","ड","ढ","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","व","श","ष","स","ह"])
        required = disclaimer_hi if contains_hindi else disclaimer_en

        if required not in response:
            # Always append divider + required line at the very end
            response = body.rstrip() + "\n---\n> \"" + required + "\"\n"
            
        # Render the final response bubble for the active chat
        assistant_placeholder.markdown(f"<span class='is-assistant'></span>{response}", unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        render_toggle(disabled=False, key_suffix="restored")

    # --- Implement Chat History Export Functionality: Provide a UI and logic for users to clear or download their chat history ---
    if st.session_state.chat_history:
        st.markdown("---")
        def confirm_clear():
            st.session_state.show_clear_confirm = True
            
        def execute_clear():
            st.session_state.chat_history = []
            st.session_state.show_clear_confirm = False
            
        def cancel_clear():
            st.session_state.show_clear_confirm = False
            
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.show_clear_confirm:
                st.warning("Are you sure you want to clear the chat?")
                sc1, sc2 = st.columns(2)
                with sc1:
                    st.button("✔️ Yes, Clear", key="yes_clear", type="primary", on_click=execute_clear)
                with sc2:
                    st.markdown("<span id='cancel-btn-highlight'></span>", unsafe_allow_html=True)
                    st.button("❌ Cancel", key="cancel_clear", on_click=cancel_clear)
            else:
                st.button("Clear Chat History", on_click=confirm_clear)
        with col2:
            export_bytes, mime_type, file_ext = _render_chat_history_txt_bytes(
                st.session_state.chat_history, 
                prediction=st.session_state.get('latest_label'),
                confidence=st.session_state.get('latest_confidence')
            )
            if export_bytes:
                st.download_button(
                    label="📥 Download Conversation History",
                    data=export_bytes,
                    file_name=f"renal_ai_chat_history.{file_ext}",
                    mime=mime_type
                )

# --- Render Application Footer: Display the developer credits at the bottom of the page ---
st.markdown(f"""
<div style="display: flex; justify-content: center; margin-top: 50px; margin-bottom: 20px;">
    <div style="background-color: {footer_bg}; color: {footer_text}; padding: 10px 30px; border-radius: 20px; font-weight: bold; box-shadow: 0 4px 8px rgba(0,0,0,0.2);">
        Made with ❤️ by Rohit.
    </div>
</div>
""", unsafe_allow_html=True)