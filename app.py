import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
from sklearn.inspection import permutation_importance

# Load model and scaler
model = joblib.load("mlp_model.pkl")
scaler = joblib.load("scaler.pkl")
X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")

# Feature names
feature_names = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_deep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen", "wifi"
]

# Page configuration
st.set_page_config(
    page_title="📱 Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #f0f0f0;
    }
    .stMarkdown, .st-b7, .st-cg, .st-cn, h1, h2, h3, h4, h5, h6 {
        color: #f0f0f0 !important;
    }
    .stNumberInput input, .stSelectbox select {
        background-color: #2c2c2c !important;
        color: #f0f0f0 !important;
        border: 1px solid #444;
    }
    .stSlider {
        color: #f0f0f0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 255, 255, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Header
colored_header(
    label="📱 Mobile Price Range Predictor",
    description="Enter your phone specifications to predict its price range",
    color_name="gray-90"
)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Phone Specifications")

    with st.expander("🔋 Battery & Hardware", expanded=True):
        battery_power = st.number_input("Battery Power (mAh)", 500, 2000, 1500)
        clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 2.0, step=0.1)
        n_cores = st.slider("Number of Cores", 1, 8, 4)
        ram = st.number_input("RAM (MB)", 256, 8000, 4000)
        int_memory = st.number_input("Internal Memory (GB)", 2, 128, 64)

    with st.expander("📷 Camera Features"):
        pc = st.slider("Primary Camera (MP)", 0, 20, 12)
        fc = st.slider("Front Camera (MP)", 0, 20, 8)

    with st.expander("📱 Display & Dimensions"):
        px_height = st.number_input("Pixel Height", 0, 1960, 1080)
        px_width = st.number_input("Pixel Width", 500, 2000, 1920)
        sc_h = st.slider("Screen Height (cm)", 5, 20, 15)
        sc_w = st.slider("Screen Width (cm)", 2, 12, 7)
        m_deep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.8, step=0.01)
        mobile_wt = st.number_input("Weight (gm)", 80, 250, 180)

    with st.expander("📶 Connectivity"):
        blue = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: "Yes" if x else "No")
        wifi = st.selectbox("WiFi", [0, 1], format_func=lambda x: "Yes" if x else "No")
        three_g = st.selectbox("3G", [0, 1], format_func=lambda x: "Yes" if x else "No")
        four_g = st.selectbox("4G", [0, 1], format_func=lambda x: "Yes" if x else "No")
        dual_sim = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "Yes" if x else "No")
        touch_screen = st.selectbox("Touch Screen", [0, 1], format_func=lambda x: "Yes" if x else "No")

    talk_time = st.slider("Talk Time (hours)", 2, 20, 10)

with col2:
    st.markdown("### 🔮 Prediction")

    if st.button("✨ Predict Price Range", use_container_width=True):
        input_data = np.array([[battery_power, blue, clock_speed, dual_sim, fc,
                                four_g, int_memory, m_deep, mobile_wt, n_cores,
                                pc, px_height, px_width, ram, sc_h, sc_w,
                                talk_time, three_g, touch_screen, wifi]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        price_map = {
            0: ("Low Cost", "#4CAF50"),
            1: ("Medium Cost", "#2196F3"),
            2: ("High Cost", "#9C27B0"),
            3: ("Very High Cost", "#FF5722")
        }

        price_text, price_color = price_map[prediction]

        card(
            title="PREDICTED PRICE RANGE",
            text=price_text,
            styles={
                "card": {
                    "width": "100%",
                    "height": "300px",
                    "border-radius": "15px",
                    "background": price_color,
                    "box-shadow": "0 10px 20px rgba(0,0,0,0.2)",
                    "color": "white",
                    "display": "flex",
                    "flex-direction": "column",
                    "justify-content": "center",
                    "align-items": "center",
                    "font-size": "32px",
                    "font-weight": "bold"
                },
                "text": {
                    "color": "white",
                    "font-size": "28px"
                },
                "title": {
                    "color": "white"
                }
            }
        )

        # Feature importance chart
        st.markdown("### 📊 Feature Importance (Permutation-Based)")
        with st.spinner("Calculating feature importance..."):
            result = permutation_importance(model, scaler.transform(X_train), y_train, n_repeats=10, random_state=42)
            importances = pd.Series(result.importances_mean, index=feature_names).sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            importances.plot(kind='barh', ax=ax, color='#66bb6a')
            ax.set_facecolor("#121212")
            fig.patch.set_facecolor("#121212")
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_title("Feature Importance", color='white')
            ax.set_xlabel("Mean Importance Score", color='white')
            st.pyplot(fig)

    else:
        card(
            title="READY TO PREDICT",
            text="Enter phone specifications and click the predict button",
            styles={
                "card": {
                    "width": "100%",
                    "height": "300px",
                    "border-radius": "15px",
                    "background": "#2c2c2c",
                    "box-shadow": "0 10px 20px rgba(255,255,255,0.05)",
                    "color": "#eeeeee",
                    "display": "flex",
                    "flex-direction": "column",
                    "justify-content": "center",
                    "align-items": "center",
                    "font-size": "24px"
                },
                "text": {
                    "color": "#eeeeee"
                },
                "title": {
                    "color": "#eeeeee"
                }
            }
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #aaaaaa; font-size: 14px;">
    <p>This machine learning model predicts mobile phone price ranges based on specifications.</p>
    <p>For best results, provide accurate information about the device.</p>
</div>
""", unsafe_allow_html=True)
