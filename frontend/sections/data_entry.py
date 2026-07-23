"""
Data Entry Page - Add, edit, and delete health records
"""

import streamlit as st
import pandas as pd
from datetime import date
from data_utils import save_data
from utils.tdee_calculator import estimate_tdee
from utils.metrics_helpers import compute_intake_targets
from utils.bulk_constants import BULK_START, PPL_DAY_MAP


def render(df):
    """Render the data entry page"""
    st.title("➕ Data Entry & Management")

    # Convert date column to datetime if it's not already
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    _render_today_target(df)

    # Main interface
    st.subheader("📅 Select or Add Date")
    
    # Date selection/input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_date = st.date_input(
            "Choose date to edit or add new record:",
            value=date.today(),
            help="Select an existing date to edit, or a new date to create a record"
        )
    
    with col2:
        # Check if date exists in data
        date_exists = False
        existing_row = None
        
        if not df.empty:
            df_date_check = df[df['date'].dt.date == selected_date]
            if len(df_date_check) > 0:
                date_exists = True
                existing_row = df_date_check.iloc[0]
                st.success("📝 Date exists - Edit mode")
            else:
                st.info("🆕 New date - Add mode")
        else:
            st.info("🆕 New date - Add mode")
    
    # Data entry form
    st.subheader("📊 Health Data Entry")
    
    with st.form("health_data_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Activity & Sleep**")
            steps = st.number_input(
                "Steps (count)", 
                min_value=0, 
                value=int(existing_row['steps']) if date_exists and pd.notna(existing_row['steps']) else 0,
                help="Daily step count"
            )
            
            sleep_min = st.number_input(
                "Sleep (minutes)", 
                min_value=0, 
                max_value=1440,
                value=int(existing_row['sleep_min']) if date_exists and pd.notna(existing_row['sleep_min']) else 0,
                help="Total sleep time in minutes"
            )
            
            move_kcal = st.number_input(
                "Move / red ring (kcal)",
                min_value=0,
                value=int(existing_row['move_kcal']) if date_exists and 'move_kcal' in df.columns and pd.notna(existing_row.get('move_kcal')) else 0,
                help="Apple Watch Move ring (active energy, 'Bouger'). Drives WADP target = 2000 + this − deficit."
            )
            # Workout duration is no longer entered here (tracked in the training app).
            # Preserve any existing value on edit so historical data is untouched.
            workout_min = int(existing_row['workout_duration_min_tot']) if date_exists and pd.notna(existing_row['workout_duration_min_tot']) else 0
        
        with col2:
            st.write("**Body & Nutrition**")
            # Default to last known weight from CSV
            _last_weight = 70.0
            if not df.empty and df['weight'].notna().any():
                _last_weight = float(df['weight'].dropna().iloc[-1])

            weight = st.number_input(
                "Weight (kg)",
                min_value=30.0,
                max_value=300.0,
                value=float(existing_row['weight']) if date_exists and pd.notna(existing_row['weight']) else _last_weight,
                step=0.1,
                help="Body weight in kilograms"
            )

            # Default to last known waist from CSV (waist_cm column may not exist on old CSVs)
            _last_waist = 0.0
            if 'waist_cm' in df.columns and df['waist_cm'].notna().any():
                _last_waist = float(df['waist_cm'].dropna().iloc[-1])

            _existing_waist = (
                float(existing_row['waist_cm'])
                if date_exists and 'waist_cm' in df.columns and pd.notna(existing_row.get('waist_cm'))
                else _last_waist
            )

            waist_cm = st.number_input(
                "Waist (cm)",
                min_value=0.0,
                max_value=200.0,
                value=_existing_waist,
                step=0.1,
                help="Waist circumference at the navel, measured morning fasted on exhale. Leave at 0 to skip.",
            )

            calories_burned = st.number_input(
                "Calories burned (kcal)", 
                min_value=0,
                value=int(existing_row['calories_burned']) if date_exists and pd.notna(existing_row['calories_burned']) else 0,
                help="Total calories burned"
            )
            
            calories_consumed = st.number_input(
                "Calories consumed (kcal)", 
                min_value=0,
                value=int(existing_row['calories_consumed']) if date_exists and pd.notna(existing_row['calories_consumed']) else 0,
                help="Total calories consumed"
            )
        
        # Form buttons
        st.markdown("---")
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
        
        with col_btn1:
            submit_btn = st.form_submit_button("💾 Save Record", type="primary", use_container_width=True)
        
        with col_btn2:
            if date_exists:
                delete_btn = st.form_submit_button("🗑️ Delete Record", type="secondary", use_container_width=True)
            else:
                delete_btn = False
        
        with col_btn3:
            clear_btn = st.form_submit_button("🧹 Clear Form", use_container_width=True)
        
        with col_btn4:
            # WADP deficit: intake target = BMR0(2000) + Move; deficit = target − consumed
            wadp_tdee = 2000 + move_kcal
            deficit = wadp_tdee - calories_consumed
            if calories_consumed == 0:
                st.info("⚖️ Enter consumed")
            elif deficit > 0:
                st.success(f"📉 WADP deficit: {deficit} kcal")
            else:
                st.error(f"📈 WADP surplus: {abs(deficit)} kcal")
    
    # Handle form submissions
    if submit_btn:
        _handle_save(df, selected_date, date_exists, steps, sleep_min, workout_min,
                    weight, calories_burned, calories_consumed, waist_cm, move_kcal)
    
    if delete_btn and date_exists:
        _handle_delete(df, selected_date)
    
    if clear_btn:
        st.rerun()

    # Science TDEE preview (non-blocking, informational)
    try:
        tdee_est = estimate_tdee(
            weight_kg=weight,
            steps=steps,
            workout_duration_min=workout_min,
            day=selected_date,
            calories_consumed=calories_consumed if calories_consumed > 0 else None,
        )
        st.info(
            f"🔬 **Science TDEE estimate (not Apple Watch):** "
            f"{tdee_est['tdee_science']} kcal — "
            f"BMR {tdee_est['bmr']} · Gym {tdee_est['eat_gym']} · "
            f"Bike {tdee_est['eat_bike']} · NEAT {tdee_est['neat']} · TEF {tdee_est['tef']}"
        )
    except Exception:
        pass  # silently skip if profile unavailable

    # Display recent data table
    _display_recent_records(df)


def _render_today_target(df):
    """Compact calorie target banner — updates automatically as new data is added."""
    if df.empty:
        return

    targets = compute_intake_targets(df, bulk_start_ts=BULK_START)
    if targets is None:
        return

    today = date.today()
    ppl_session = PPL_DAY_MAP.get(today.weekday(), "Training")
    is_rest_day = ppl_session == "Rest"

    # Check if today has a logged workout to override the schedule guess
    today_row = df[df['date'].dt.date == today] if not df.empty else pd.DataFrame()
    if not today_row.empty and pd.notna(today_row.iloc[0].get('workout_duration_min_tot')):
        is_training = today_row.iloc[0]['workout_duration_min_tot'] > 30
        day_label = ppl_session if is_training else "Rest day"
        day_kcal = targets['training_day'] if is_training else targets['rest_day']
        color = "#6366f1" if is_training else "#94a3b8"
        st.markdown(
            f"""
            <div style="
                background:#1e293b; border-left:4px solid {color};
                border-radius:8px; padding:0.65rem 1rem;
                display:flex; align-items:center; gap:1.5rem; margin-bottom:1rem;
            ">
                <span style="color:#9ca3af; font-size:0.8rem;">Today's target ({day_label})</span>
                <span style="color:{color}; font-size:1.25rem; font-weight:700;">{day_kcal:,} kcal</span>
                <span style="color:#6b7280; font-size:0.75rem;">
                    🏋️ Training: {targets['training_day']:,} &nbsp;|&nbsp; 🛋️ Rest: {targets['rest_day']:,}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # No logged data yet — use PPL schedule to suggest today's target
        day_kcal = targets['rest_day'] if is_rest_day else targets['training_day']
        color = "#94a3b8" if is_rest_day else "#6366f1"
        st.markdown(
            f"""
            <div style="
                background:#1e293b; border-left:4px solid {color};
                border-radius:8px; padding:0.65rem 1rem;
                display:flex; align-items:center; gap:2rem; margin-bottom:1rem;
            ">
                <span style="color:#9ca3af; font-size:0.8rem;">Today ({ppl_session})</span>
                <span style="color:{color}; font-size:1.25rem; font-weight:700;">{day_kcal:,} kcal</span>
                <span style="color:#6b7280; font-size:0.75rem;">
                    🏋️ Training: {targets['training_day']:,} &nbsp;|&nbsp;
                    🛋️ Rest: {targets['rest_day']:,} &nbsp;|&nbsp;
                    TDEE ~{targets['empirical_tdee']:,}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _handle_save(df, selected_date, date_exists, steps, sleep_min, workout_min,
                 weight, calories_burned, calories_consumed, waist_cm=0.0, move_kcal=0):
    """Handle saving a record"""
    new_record = {
        'user_id': '',
        'date': pd.Timestamp(selected_date),
        'steps': steps if steps > 0 else None,
        'sleep_min': sleep_min if sleep_min > 0 else None,
        'workout_duration_min_tot': workout_min if workout_min > 0 else None,
        'weight': weight if weight > 0 else None,
        'calories_burned': calories_burned if calories_burned > 0 else None,
        'calories_consumed': calories_consumed if calories_consumed > 0 else None,
        'waist_cm': waist_cm if waist_cm > 0 else None,
        'move_kcal': move_kcal if move_kcal > 0 else None,
    }

    # Make sure the df has waist_cm / move_kcal columns before updating (legacy CSVs)
    if 'waist_cm' not in df.columns:
        df['waist_cm'] = pd.NA
    if 'move_kcal' not in df.columns:
        df['move_kcal'] = pd.NA

    try:
        if date_exists:
            # Update existing record
            df.loc[df['date'].dt.date == selected_date, 'steps'] = new_record['steps']
            df.loc[df['date'].dt.date == selected_date, 'sleep_min'] = new_record['sleep_min']
            df.loc[df['date'].dt.date == selected_date, 'workout_duration_min_tot'] = new_record['workout_duration_min_tot']
            df.loc[df['date'].dt.date == selected_date, 'weight'] = new_record['weight']
            df.loc[df['date'].dt.date == selected_date, 'calories_burned'] = new_record['calories_burned']
            df.loc[df['date'].dt.date == selected_date, 'calories_consumed'] = new_record['calories_consumed']
            df.loc[df['date'].dt.date == selected_date, 'waist_cm'] = new_record['waist_cm']
            df.loc[df['date'].dt.date == selected_date, 'move_kcal'] = new_record['move_kcal']

            save_data(df, st.session_state.csv_path)
            st.success(f"✅ Record for {selected_date} updated successfully!")
        else:
            # Add new record
            new_row_df = pd.DataFrame([new_record])
            df_updated = pd.concat([df, new_row_df], ignore_index=True)
            df_updated = df_updated.sort_values('date')
            
            save_data(df_updated, st.session_state.csv_path)
            st.success(f"✅ New record for {selected_date} added successfully!")
        
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error saving record: {str(e)}")


def _handle_delete(df, selected_date):
    """Handle deleting a record"""
    try:
        df_updated = df[df['date'].dt.date != selected_date].copy()
        save_data(df_updated, st.session_state.csv_path)
        st.success(f"🗑️ Record for {selected_date} deleted successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Error deleting record: {str(e)}")


def _display_recent_records(df):
    """Display table of recent records"""
    st.markdown("---")
    st.subheader("📋 Recent Records")
    
    if df.empty:
        st.info("No records available. Add your first record above!")
    else:
        # Show last 10 records
        display_df = df.sort_values('date', ascending=False).head(10).copy()
        
        # Format the data for display
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        
        # Ensure waist_cm column exists for display (legacy CSVs)
        if 'waist_cm' not in display_df.columns:
            display_df['waist_cm'] = pd.NA

        # Format numeric columns
        numeric_columns = ['steps', 'sleep_min', 'workout_duration_min_tot', 'weight',
                          'calories_burned', 'calories_consumed', 'waist_cm']
        for col in numeric_columns:
            if col in ('weight', 'waist_cm'):
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")

        # Display the table
        st.dataframe(
            display_df[['date', 'steps', 'sleep_min', 'workout_duration_min_tot',
                       'weight', 'waist_cm', 'calories_burned', 'calories_consumed']],
            use_container_width=True,
            column_config={
                'date': st.column_config.TextColumn('Date', width='medium'),
                'steps': st.column_config.TextColumn('Steps', width='small'),
                'sleep_min': st.column_config.TextColumn('Sleep (min)', width='small'),
                'workout_duration_min_tot': st.column_config.TextColumn('Workout (min)', width='small'),
                'weight': st.column_config.TextColumn('Weight (kg)', width='small'),
                'waist_cm': st.column_config.TextColumn('Waist (cm)', width='small'),
                'calories_burned': st.column_config.TextColumn('Cal. Burned', width='small'),
                'calories_consumed': st.column_config.TextColumn('Cal. Consumed', width='small')
            },
            hide_index=True
        )
        
        # Quick summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_records = len(df)
            st.metric("📊 Total Records", total_records)