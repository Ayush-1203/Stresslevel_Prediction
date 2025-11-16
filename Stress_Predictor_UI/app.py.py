import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
import time

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Stress Analyzer", page_icon="🧠", layout="wide")

# ---------------------------
# Custom CSS (Professional, Stylish, Icons)
# ---------------------------
st.markdown("""
<style>
/* 1. MAIN BACKGROUND & FONT */
.stApp {
    background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* 2. BACKGROUND ICONS (Watermark effect) */
.stApp::before {
    content: "🧠 💡 🧠 💡 🧠 💡 ";
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 150px;
    color: rgba(255, 255, 255, 0.1); /* Very transparent */
    z-index: 0;
    pointer-events: none;
    white-space: nowrap;
}

/* 4. STYLISH WHITE TITLE */
.big-title {
    font-family: 'Playfair Display', serif; /* Professional Serif */
    font-size: 60px; 
    font-weight: 900;
    color: #ffffff; /* White as requested */
    text-align: center;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    margin-bottom: 5px;
    letter-spacing: 1px;
}

.subtitle {
    text-align: center;
    color: #f0f0f0;
    margin-bottom: 30px;
    font-size: 25px;
    font-weight: 500;
}

/* 5. INPUT FIELDS - WHITE BACKGROUND */
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 20px !important;
    border-radius: 8px;
    border: 1px solid #ddd;
}


/* 6. LABELS - BLACK & BIGGER */
.stSelectbox label, .stTextInput label {
    color: #000000 !important;
    font-size: 20px !important; /* Increased size */
    font-weight: 700 !important;
}

/* 7. BUTTONS - PROFESSIONAL WHITE */
.stButton > button {
    background-color: #ffffff;
    color: #185a9d;
    font-size: 20px;
    font-weight: bold;
    padding: 12px 28px;
    border-radius: 30px;
    border: 2px solid transparent;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    background-color: #f8f9fa;
}

/* 8. CARDS & CONTAINERS */
.card {
    background: rgba(255, 255, 255, 0.85); /* Slightly opaque white for readability */
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    border: 1px solid rgba(255, 255, 255, 0.18);
    margin-bottom: 20px;
}

/* 9. CENTERED QUOTE */
.quote-container {
    text-align: center;
    margin: 20px auto;
    padding: 15px;
    max-width: 800px;
}
.quote-text {
    font-family: 'Georgia', serif;
    font-style: italic;
    font-size: 24px;
    color: #000000; /* Black text */
    font-weight: 600;
}
.quote-author {
    font-size: 18px;
    color: #333;
    margin-top: 5px;
}



/* Hide default streamlit menu/footer for cleaner look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Files & Data Init (20 Features)
# ---------------------------
USERS_CSV = "users.csv"
HISTORY_CSV = "history.csv"

# Check Users File
if not os.path.exists(USERS_CSV):
    pd.DataFrame({"username": ["Ayush"], "password": ["1234"]}).to_csv(USERS_CSV, index=False)

# Check History File (Updated with 20 features)
if not os.path.exists(HISTORY_CSV):
    cols = [
        "username","timestamp","email","stress_level",
        "anxiety_level","self_esteem","mental_health_history","depression","headache",
        "blood_pressure","sleep_quality","breathing_problem",
        "living_conditions","safety","basic_needs","academic_performance",
        "study_load","teacher_student_relationship","future_career_concerns",
        "social_support","peer_pressure","extracurricular_activities",
        "screen_time", "health_issues" # <-- NEW COLUMNS
    ]
    pd.DataFrame(columns=cols).to_csv(HISTORY_CSV, index=False)

# ---------------------------
# Helper Functions
# ---------------------------
def load_users(): return pd.read_csv(USERS_CSV)
def save_users(df): df.to_csv(USERS_CSV, index=False)
def load_history(): return pd.read_csv(HISTORY_CSV)
def save_history(df): df.to_csv(HISTORY_CSV, index=False)

# ---------------------------
# Session State
# ---------------------------
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "username" not in st.session_state: st.session_state.username = None

# ===================================================================
# LOGIN PAGE
# ===================================================================
if not st.session_state.logged_in:
    st.markdown("<div class='login-wrapper'>", unsafe_allow_html=True)
    
    # Title
    st.markdown("<div class='big-title'>Stress Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Professional Mental Well-being Predictor</div>", unsafe_allow_html=True)
    
    # Quote
    st.markdown("""
    <div class='quote-container'>
        <div class='quote-text'>"The greatest weapon against stress is our ability to choose one thought over another."</div>
        <div class='quote-author'>– William James</div>
    </div>
    """, unsafe_allow_html=True)

    # Login Card
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    
    tab = st.radio("", ("Login", "Register"), horizontal=True)
    st.markdown("---")

    if tab == "Login":
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            users = load_users()
            if not users.empty and (users["username"].str.lower() == u.lower()).any():
                stored_pw = users.loc[users["username"].str.lower() == u.lower(), "password"].values[0]
                if str(stored_pw) == p:
                    st.session_state.logged_in = True
                    st.session_state.username = u
                    st.rerun()
                else:
                    st.error("Incorrect Password")
            else:
                st.error("User not found")

    else: # Register
        new_u = st.text_input("New Username", key="reg_user")
        new_p = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            users = load_users()
            if (users["username"].str.lower() == new_u.lower()).any():
                st.error("Username taken")
            elif new_u and new_p:
                new_user = pd.DataFrame([[new_u, new_p]], columns=["username", "password"])
                users = pd.concat([users, new_user], ignore_index=True)
                save_users(users)
                st.success("Registered! Please Login.")

    st.markdown("</div>", unsafe_allow_html=True) # End Card
    st.markdown("</div>", unsafe_allow_html=True) # End Wrapper
    st.stop()

# ===================================================================
# MAIN DASHBOARD
# ===================================================================
username = st.session_state.username

# Header
c1, c2 = st.columns([8, 2])
with c1:
    st.markdown(f"<h2 style='color:grey; margin:0; padding-top:10px;'>Welcome, {username} 👋</h2>", unsafe_allow_html=True)
with c2:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# Stylish Title & Quote
st.markdown("<div class='big-title'>Stress Analyzer</div>", unsafe_allow_html=True)
st.markdown("""
<div class='quote-container'>
    <div class='quote-text'>"It’s not the load that breaks you down, it’s the way you carry it."</div>
</div>
""", unsafe_allow_html=True)

# Main Form Card
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3 style='color:black; text-align:center; margin-bottom:30px;'>🧾 Enter Your Health Parameters</h3>", unsafe_allow_html=True)

# Load Model
try:
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("⚠ Models not found. Please ensure 'best_model.pkl' and 'scaler.pkl' are in the folder.")
    st.stop()

# Input Grid (20 Features)
def input_box(label, key):
    return st.selectbox(label, list(range(1, 11)), key=key)

c1, c2, c3, c4 = st.columns(4) # Changed to 4 columns to fit 20 items neatly

with c1:
    anxiety = input_box("Anxiety (1-10)", "anx")
    self_esteem = input_box("Self Esteem (1-10)", "self")
    mental_hist = st.selectbox("Mental History (0/1)", [0,1], key="mh")
    depression = st.selectbox("Depression (0/1)", [0,1], key="dep")
    headache = st.selectbox("Headache (0/1)", [0,1], key="head")

with c2:
    bp = input_box("Blood Pressure (1-10)", "bp")
    sleep = input_box("Sleep Quality (1-10)", "sleep")
    breathing = st.selectbox("Breathing Prob (0/1)", [0,1], key="bre")
    living = input_box("Living Cond (1-10)", "live")
    safety = input_box("Safety (1-10)", "safe")

with c3:
    needs = input_box("Basic Needs (1-10)", "needs")
    academic = input_box("Academic Perf (1-10)", "acad")
    load = input_box("Study Load (1-10)", "load")
    teacher = input_box("Teacher Rel (1-10)", "teacher")
    career = input_box("Career Concern (1-10)", "career")

with c4:
    support = input_box("Social Support (1-10)", "social")
    peer = input_box("Peer Pressure (1-10)", "peer")
    extra = input_box("Extracurricular (1-10)", "extra")
    # --- NEW FEATURES ---
    screen_time = input_box("Screen Time (1-10)", "screen")
    health_issues = st.selectbox("Health Issues (0/1)", [0,1], key="health")

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
if st.button("🔍 Analyze Stress Level"):
    # Array of 20 features
    features = np.array([[
        anxiety, self_esteem, mental_hist, depression, headache,
        bp, sleep, breathing, living, safety, 
        needs, academic, load, teacher, career, 
        support, peer, extra, screen_time, health_issues
    ]])

    try:
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        
        # Save Result
        new_entry = pd.DataFrame([{
            "username": username,
            "timestamp": datetime.now().timestamp(),
            "email": "",
            "stress_level": prediction,
            "anxiety_level": anxiety, "self_esteem": self_esteem, "mental_health_history": mental_hist,
            "depression": depression, "headache": headache, "blood_pressure": bp,
            "sleep_quality": sleep, "breathing_problem": breathing, "living_conditions": living,
            "safety": safety, "basic_needs": needs, "academic_performance": academic,
            "study_load": load, "teacher_student_relationship": teacher, "future_career_concerns": career,
            "social_support": support, "peer_pressure": peer, "extracurricular_activities": extra,
            "screen_time": screen_time, "health_issues": health_issues
        }])
        
        hist = load_history()
        hist = pd.concat([hist, new_entry], ignore_index=True)
        save_history(hist)

        # Display Result
        st.markdown("---")
        st.markdown("<h2 style='text-align:center; color:black;'>Prediction Result</h2>", unsafe_allow_html=True)
        
        if prediction == 0:
            st.success("🌟 LOW STRESS: You are doing great! Keep maintaining this balance,Practice gratitude and help others.")
            st.balloons()
        elif prediction == 1:
            st.warning("⚠ MODERATE STRESS: You are feeling the pressure. Take short breaks, Priortize & delegate, move your body.")
        else:
            st.error("🚨 HIGH STRESS: Immediate attention needed. Please prioritize rest , Talk to someone , Seek support.")

    except ValueError as e:
        st.error(f"❌ Model Dimension Error: The loaded model expects different features than the 20 provided. ({str(e)})")
        st.info("Please retrain your model with the new features: Screen Time & Health Issues.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.markdown("</div>",unsafe_allow_html=True)# End Card