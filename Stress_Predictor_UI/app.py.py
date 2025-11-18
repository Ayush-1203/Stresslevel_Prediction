# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import hashlib
from datetime import datetime
import time

# Machine learning imports for fallback training
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Visualization
import altair as alt

# ---------------------------
# Config & Files
# ---------------------------
st.set_page_config(page_title="Stress Analyzer", page_icon="🧠", layout="wide")
USERS_CSV = "users.csv"
HISTORY_CSV = "history.csv"
MODEL_FILE = "best_model.pkl"
SCALER_FILE = "scaler.pkl"

# Columns (20 feature set + metadata)
FEATURE_COLUMNS = [
    "anxiety_level","self_esteem","mental_health_history","depression","headache",
    "blood_pressure","sleep_quality","breathing_problem","living_conditions","safety",
    "basic_needs","academic_performance","study_load","teacher_student_relationship","future_career_concerns",
    "social_support","peer_pressure","extracurricular_activities","screen_time","health_issues"
]

HISTORY_COLUMNS = ["username","timestamp","dt_iso","email","stress_level"] + FEATURE_COLUMNS

# ---------------------------
# Utility helpers
# ---------------------------
def sha256_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_files_exist():
    if not os.path.exists(USERS_CSV):
        df = pd.DataFrame([{"username":"Ayush","password":sha256_hash("1234"), "role":"user"}])
        df.to_csv(USERS_CSV, index=False)
    if not os.path.exists(HISTORY_CSV):
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(HISTORY_CSV, index=False)

def load_users():
    return pd.read_csv(USERS_CSV)

def save_users(df):
    df.to_csv(USERS_CSV, index=False)

def load_history():
    return pd.read_csv(HISTORY_CSV)

def save_history(df):
    df.to_csv(HISTORY_CSV, index=False)

# Create fallback model and scaler if missing
def create_and_save_fallback_model():
    X = np.random.rand(1000, len(FEATURE_COLUMNS)) * 9 + 1  
    y = np.random.choice([0,1,2], size=X.shape[0], p=[0.5,0.35,0.15])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(Xs, y)
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return model, scaler

def load_model_and_scaler():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            scaler = joblib.load(SCALER_FILE)
            return model, scaler, "loaded"
        except Exception:
            return create_and_save_fallback_model()
    else:
        return create_and_save_fallback_model()

# ---------------------------
# App Initialization
# ---------------------------
ensure_files_exist()
model, scaler, model_status = load_model_and_scaler()

# ---------------------------
# STYLING: Animated gradient, floating icons, neon buttons, large fonts
# ---------------------------
st.markdown("""
<style>

    /* ================================
       BLUE LAGOON BACKGROUND THEME
       ( #43C6AC → #191654 )
    ================================== */
    .stApp {
        background: linear-gradient(135deg, #43C6AC 0%, #191654 100%);
        background-attachment: fixed;
        background-size: cover;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        animation: fadeInBG 1.5s ease-in-out;
    }

    @keyframes fadeInBG {
        from { opacity: 0; }
        to   { opacity: 1; }
    }

    /* =================================
        GLOBAL TEXT SIZE UPGRADE
       (Safe: DOES NOT affect Analyzer UI
    ================================== */
    h1, h2, h3, h4, h5 {
        color: #ffffff !important;
        font-size: 60px !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
    }

    p, div, span, label {
        font-size: 30px !important;
        color: #f6f6f6 !important;
    }

    /* =============== CARDS =============== */
    .card {
        background: rgba(255,255,255,0.10) !important;
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 6px 26px rgba(0,0,0,0.35);
        padding: 30px !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: 0.3s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 36px rgba(0,0,0,0.45);
    }

    /* ============= SIDEBAR ============== */
    .stSidebar {
        background: rgba(25,25,50,0.35) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255,255,255,0.12);
    }

    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        font-size: 40px !important;
        color: white !important;
    }

    /* ============= BUTTONS ============== */
    .stButton>button {
        background: linear-gradient(90deg, #00f2ff, #0061ff) !important;
        color: white !important;
        font-size: 30px !important;
        padding: 14px 34px !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 0 18px rgba(0,170,255,0.55);
        transition: .25s ease-in-out;
    }

    .stButton>button:hover {
        transform: scale(1.06);
        box-shadow: 0 0 30px rgba(0,220,255,0.75);
    }

    /* ============= INPUTS (SAFE) ============= */
    .stTextInput>div>div>input {
        background: rgba(255,255,255,0.15) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
    }

    .stSelectbox div[data-baseweb="select"] {
        background: rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: white !important;
    }

    /* =========== CHART BOXES =========== */
    .vega-embed {
        background: rgba(255,255,255,0.08) !important;
        padding: 20px !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 28px rgba(0,0,0,0.35) !important;
    }
            /* ---------------------------
       REMOVE EXTRA BOX UNDER TITLE
       --------------------------- */
    .block-container > div:nth-child(2) {
        display: none !important;
    }

    /* ---------------------------
       MAIN TITLE: Stress Analyzer
       --------------------------- */
    .home-title {
        font-size: 500px !important;
        font-weight: 900 !important;
        text-align: center !important;
        color: #ffffff !important;
        letter-spacing: 1.5px;
        margin-top: 40px !important;
        margin-bottom: 10px !important;
        text-shadow: 0 5px 30px rgba(0,0,0,0.45);
    }

    /* ---------------------------
       SUBTITLE BELOW TITLE
       --------------------------- */
    .home-subtitle {
        font-size: 32px !important;
        font-weight: 500 !important;
        text-align: center !important;
        color: #dff9ff !important;
        margin-top: -5px !important;
        margin-bottom: 25px !important;
        line-height: 1.5;
    }
            
            

    

</style>
""", unsafe_allow_html=True)



# # Inject floating brain / bulb images (uses fixed positioned img tags)
# st.markdown(
#     """
#     <div>
#       <img class="floating-icon b1" src="https://cdn-icons-png.flaticon.com/512/4140/4140037.png" />
#       <img class="floating-icon b2" src="https://cdn-icons-png.flaticon.com/512/3037/3037129.png" />
#       <img class="floating-icon b3" src="https://cdn-icons-png.flaticon.com/512/2995/2995604.png" />
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Stress Analyzer")
nav = st.sidebar.radio("Navigate", ("Home","Analyze","History","Admin","About"))

# ---------------------------
# Authentication
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = "user"

def auth_box():
    st.sidebar.subheader("Account")
    tab = st.sidebar.selectbox("", ["Login","Register"])
    if tab == "Login":
        u = st.sidebar.text_input("Username", key="login_user")
        p = st.sidebar.text_input("Password", type="password", key="login_pass")
        if st.sidebar.button("Login"):
            users = load_users()
            if not users.empty and (users["username"].str.lower() == u.lower()).any():
                stored_pw = users.loc[users["username"].str.lower() == u.lower(), "password"].values[0]
                if str(stored_pw) == sha256_hash(p):
                    st.session_state.logged_in = True
                    st.session_state.username = users.loc[
                        users["username"].str.lower() == u.lower(), "username"
                    ].values[0]
                    st.session_state.role = users.loc[
                        users["username"].str.lower() == u.lower(), "role"
                    ].values[0] if "role" in users.columns else "user"
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Incorrect Password")
            else:
                st.sidebar.error("User not found")
    else:
        new_u = st.sidebar.text_input("New Username")
        new_p = st.sidebar.text_input("New Password", type="password")
        if st.sidebar.button("Register"):
            if not new_u or not new_p:
                st.sidebar.error("Please fill username and password")
            else:
                users = load_users()
                if (users["username"].str.lower() == new_u.lower()).any():
                    st.sidebar.error("Username taken")
                else:
                    new_row = {"username": new_u, "password": sha256_hash(new_p), "role": "user"}
                    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
                    save_users(users)
                    st.sidebar.success("Registered! Please login.")

if not st.session_state.logged_in:
    auth_box()
else:
    with st.sidebar:
        st.write(f"*Logged in as:* {st.session_state.username} ({st.session_state.role})")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.session_state.role = "user"
            st.experimental_rerun()

# ---------------------------
# NAV: Home
# ---------------------------
if nav == "Home":
    st.markdown("<h1 class='home-title'>Stress Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='home-subtitle'>Your mind is your greatest ally—nurture it, and every challenge becomes a stepping stone.</p>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    
    st.markdown("---")
    st.markdown("*Welcome to Stress Analyzer !!💡*")
    st.write(" - Your personal space to understand, track, and transform your mental well-being.😊")
    st.write("- In just a few minutes, you’ll gain clear insights into your stress patterns and receive guidance that helps you reclaim your calm, focus, and confidence.📊")
    st.write("- This is more than a tool — it’s your companion on the journey toward a healthier, stronger you💪.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# NAV: Analyze
# ---------------------------
if nav == "Analyze":
    if not st.session_state.logged_in:
        st.info("Please login to start analysis.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.header("Enter your parameters")

        def input_select(label, key, minv=1, maxv=10):
            return st.slider(label, min_value=minv, max_value=maxv,
                             value=int((minv+maxv)//2), key=key)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            anxiety = input_select("Anxiety (1-10)", "anx")
            self_esteem = input_select("Self Esteem (1-10)", "self")
            mental_hist = st.selectbox("Mental Health History (0/1)", [0,1], key="mh")
            depression = st.selectbox("Depression (0/1)", [0,1], key="dep")
            headache = st.selectbox("Headache (0/1)", [0,1], key="head")

        with c2:
            bp = input_select("Blood Pressure (1-10)", "bp")
            sleep = input_select("Sleep Quality (1-10)", "sleep")
            breathing = st.selectbox("Breathing Problem (0/1)", [0,1], key="bre")
            living = input_select("Living Conditions (1-10)", "live")
            safety = input_select("Safety (1-10)", "safe")

        with c3:
            needs = input_select("Basic Needs (1-10)", "needs")
            academic = input_select("Academic Performance (1-10)", "acad")
            load = input_select("Study Load (1-10)", "load")
            teacher = input_select("Teacher-Student Relationship (1-10)", "teacher")
            career = input_select("Future Career Concerns (1-10)", "career")

        with c4:
            support = input_select("Social Support (1-10)", "social")
            peer = input_select("Peer Pressure (1-10)", "peer")
            extra = input_select("Extracurricular (1-10)", "extra")
            screen_time = input_select("Screen Time (1-10)", "screen")
            health_issues = st.selectbox("Health Issues (0/1)", [0,1], key="health")

        if st.button("🔍 Analyze Stress Level"):

            features = np.array([[anxiety, self_esteem, mental_hist, depression, headache,
                                  bp, sleep, breathing, living, safety, needs, academic,
                                  load, teacher, career, support, peer, extra,
                                  screen_time, health_issues]])

            try:
                scaled = scaler.transform(features)
                pred = model.predict(scaled)[0]

                label_map = {0:("LOW","🌟"), 1:("MODERATE","⚠"), 2:("HIGH","🚨")}
                tag, emoji = label_map.get(pred, ("UNKNOWN","❔"))

                # Save to history
                hist = load_history()
                now = datetime.now()
                new_row = {
                    "username": st.session_state.username,
                    "timestamp": now.timestamp(),
                    "dt_iso": now.isoformat(),
                    "email": "",
                    "stress_level": int(pred)
                }
                for i,col in enumerate(FEATURE_COLUMNS):
                    new_row[col] = int(features[0][i])

                hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
                save_history(hist)

                st.success(f"{emoji} Predicted: {tag} stress")

                if pred == 0:
                    st.info("You're doing well — keep it up!")
                elif pred == 1:
                    st.warning("Moderate stress — take breaks and manage priorities.")
                else:
                    st.error("High stress detected — seek help or talk to someone you trust.")

                if hasattr(model, "feature_importances_"):
                    fi = model.feature_importances_
                    fi_df = pd.DataFrame({
                        "feature": FEATURE_COLUMNS,
                        "importance": fi
                    }).sort_values("importance", ascending=False)
                    st.markdown("#### Top features influencing model")
                    st.table(fi_df.head(6))

            except Exception as e:
                st.error(f"Error during prediction: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# NAV: History (REDESIGNED ANALYTICS DASHBOARD)
# ---------------------------
if nav == "History":
    if not st.session_state.logged_in:
        st.info("Login to view your history.")
    else:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.header("Submission History & Analytics")

        hist = load_history()
        if hist.empty:
            st.info("No history yet.")
        else:
            # Convert dt_iso to datetime safely
            hist["dt_iso"] = pd.to_datetime(hist["dt_iso"], errors="coerce")
            # Latest entries for user
            user_hist = hist[hist["username"] == st.session_state.username].copy()
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Your recent entries")
                if not user_hist.empty:
                    st.dataframe(user_hist.sort_values("timestamp", ascending=False).head(20))
                    csv = user_hist.to_csv(index=False).encode("utf-8")
                    st.download_button("Download your history (CSV)", csv, file_name=f"{st.session_state.username}_history.csv")
                else:
                    st.info("You don't have personal entries yet.")

            with col2:
                st.subheader("Quick stats")
                total = len(hist)
                unique_users = hist["username"].nunique()
                last_week = hist[hist["dt_iso"] >= (pd.Timestamp.now() - pd.Timedelta(days=7))]
                st.metric("Total entries", total)
                st.metric("Unique users", unique_users)
                st.metric("Entries last 7d", len(last_week))

            st.markdown("---")

            # Analytics cards
            st.subheader("Stress distribution")
            dist = hist["stress_level"].value_counts().sort_index().reset_index()
            dist.columns = ["stress_level", "count"]
            dist["label"] = dist["stress_level"].map({0:"Low",1:"Moderate",2:"High"})
            bar = alt.Chart(dist).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X('label:N', sort=["Low","Moderate","High"], title="Stress Level"),
                y=alt.Y('count:Q', title="Count"),
                color=alt.Color('label:N', scale=alt.Scale(domain=["Low","Moderate","High"], range=["#56CCF2","#FEC260","#FF6A88"])),
                tooltip=['label','count']
            ).properties(width=600, height=360)
            st.altair_chart(bar, use_container_width=True)

            st.markdown("### Trend over time")
            recent = hist.set_index("dt_iso").resample("7D")["stress_level"].mean().reset_index()
            if not recent.empty:
                line = alt.Chart(recent).mark_line(point=True).encode(
                    x=alt.X('dt_iso:T', title='Date'),
                    y=alt.Y('stress_level:Q', title='Average stress (0-2)'),
                    tooltip=[alt.Tooltip('dt_iso:T', title='Date'), alt.Tooltip('stress_level:Q', title='Avg stress')]
                ).properties(width=800, height=320)
                area = alt.Chart(recent).mark_area(opacity=0.1).encode(
                    x='dt_iso:T', y='stress_level:Q'
                )
                st.altair_chart(area + line, use_container_width=True)
            else:
                st.info("Not enough data to show trend.")

            st.markdown("### Stress share (pie)")
            pie_df = dist.copy()
            if not pie_df.empty:
                pie = alt.Chart(pie_df).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="count", type="quantitative"),
                    color=alt.Color(field="label", type="nominal", scale=alt.Scale(range=["#56CCF2","#FEC260","#FF6A88"])),
                    tooltip=['label','count']
                ).properties(width=350, height=350)
                st.altair_chart(pie, use_container_width=False)
            else:
                st.info("No distribution yet.")

            st.markdown("---")
            st.subheader("Latest site entries")
            st.dataframe(hist.sort_values("timestamp", ascending=False).head(50))

            # Admin-only full download
            if st.session_state.role == "admin":
                csv_all = hist.to_csv(index=False).encode("utf-8")
                st.download_button("Download full history (CSV)", csv_all, file_name="history_full.csv")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# NAV: Admin
# ---------------------------
if nav == "Admin":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Admin Panel")

    if st.session_state.role != "admin":
        st.warning("Admin-only panel.")
        users = load_users()
        st.markdown("Promote a user (demo only)")
        sel = st.selectbox("User", users["username"].tolist())
        if st.button("Promote to admin"):
            users.loc[users["username"]==sel, "role"]="admin"
            save_users(users)
            st.success("User promoted.")
    else:
        st.success("Welcome, admin 👑")
        users = load_users()
        st.dataframe(users)

        if st.button("Export users"):
            st.download_button("Download users.csv", users.to_csv(index=False).encode("utf-8"))

        upload = st.file_uploader("Upload model (.pkl)", type=["pkl","joblib"])
        if upload:
            with open(MODEL_FILE, "wb") as f:
                f.write(upload.getbuffer())
            st.success("Model uploaded!")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# NAV: About
# ---------------------------
if nav == "About":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("About")
    st.write("""
    This application uses machine learning to estimate stress levels based on 20 input features.
    It includes:
    - Authentication  
    - Real-time predictions  
    - Redesigned analytics dashboard with colorful charts  
    - Animated background and floating icons  
    - Neon-glow action buttons
    """)
    st.markdown("</div>", unsafe_allow_html=True)
