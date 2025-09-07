import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date

st.set_page_config(page_title="Personal Health Dashboard", layout="wide")

DATA_PATH = "data/sample_daily.csv"

# --- Load CSV ---
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

df = load_data()

# --- Sidebar: Upload new CSV ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["date"])
    df.sort_values("date", inplace=True)

# --- Sidebar: Manual Entry Form ---
st.sidebar.header("Manual Entry")
with st.sidebar.form("manual_entry_form", clear_on_submit=True):
    st.write("Add a new daily record:")
    entry_date = st.date_input("Date", value=date.today())
    source = st.selectbox("Source", ["manual", "apple_watch", "nutrition_app"])
    steps = st.number_input("Steps", min_value=0, step=100, value=0)
    calories_burned = st.number_input("Calories Burned", min_value=0, step=10, value=0)
    calories_consumed = st.number_input("Calories Consumed", min_value=0, step=10, value=0)
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    weight_kg = st.number_input("Weight (kg)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    workout_count = st.number_input("Workout Count", min_value=0, step=1, value=0)
    workout_duration_total_minutes = st.number_input("Workout Duration (min)", min_value=0, step=5, value=0)
    notes = st.text_area("Notes", "")
    submitted = st.form_submit_button("➕ Add Entry")

    if submitted:
        new_row = {
            "user_id": "manual_user",  # placeholder, adapt if you manage multiple users
            "date": pd.to_datetime(entry_date),
            "source": source,
            "steps": steps,
            "calories_burned": calories_burned,
            "calories_consumed": calories_consumed,
            "sleep_hours": sleep_hours,
            "weight_kg": weight_kg,
            "workout_count": workout_count,
            "workout_duration_totel_minutes": workout_duration_total_minutes,
            "notes": notes,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.sort_values("date", inplace=True)

        # Save back to CSV so it persists between runs
        df.to_csv(DATA_PATH, index=False)
        st.success(f"Entry for {entry_date} added!")

# --- Title ---
st.title("🏋️ Personal Health & Fitness Dashboard")

# --- KPIs ---
latest = df.iloc[-1]
last_7d = df.tail(7).mean(numeric_only=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Steps (7d avg)", f"{int(last_7d['steps']):,}", f"Last: {latest['steps']:,}")
col2.metric("Calories Burned (7d avg)", f"{int(last_7d['calories_burned'])}", f"Last: {latest['calories_burned']}")
col3.metric("Sleep (7d avg hrs)", f"{last_7d['sleep_hours']:.1f}", f"Last: {latest['sleep_hours']}h")
col4.metric("Weight (kg)", f"{latest['weight_kg']:.1f}", f"{latest['weight_kg']-df.iloc[-2]['weight_kg']:.1f} vs prev")

# --- Charts ---
st.subheader("📈 Trends Over Time")

metrics = [
    "steps", "calories_burned", "calories_consumed",
    "sleep_hours", "weight_kg", "workout_count",
    "workout_duration_totel_minutes"
]

for metric in metrics:
    fig = px.line(df, x="date", y=metric, color="source",
                  markers=True, title=f"{metric.replace('_',' ').title()} over Time")
    st.plotly_chart(fig, use_container_width=True)

# --- Raw Data ---
st.subheader("📊 Raw Data")
st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)