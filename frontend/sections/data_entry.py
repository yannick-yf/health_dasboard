"""
Data Entry Page - Add, edit, and delete health records
"""

import streamlit as st
import pandas as pd
from datetime import date
from data_utils import save_data


def render(df):
    """Render the data entry page"""
    st.title("➕ Data Entry & Management")
    
    # Convert date column to datetime if it's not already
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
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
            
            workout_min = st.number_input(
                "Workout duration (minutes)", 
                min_value=0,
                value=int(existing_row['workout_duration_min_tot']) if date_exists and pd.notna(existing_row['workout_duration_min_tot']) else 0,
                help="Total workout time in minutes"
            )
        
        with col2:
            st.write("**Body & Nutrition**")
            weight = st.number_input(
                "Weight (kg)", 
                min_value=30.0, 
                max_value=300.0,
                value=float(existing_row['weight']) if date_exists and pd.notna(existing_row['weight']) else 70.0,
                step=0.1,
                help="Body weight in kilograms"
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
            # Show calorie balance
            balance = calories_burned - calories_consumed
            if balance > 0:
                st.success(f"📉 Deficit: {balance} kcal")
            elif balance < 0:
                st.error(f"📈 Surplus: {abs(balance)} kcal")
            else:
                st.info("⚖️ Balanced")
    
    # Handle form submissions
    if submit_btn:
        _handle_save(df, selected_date, date_exists, steps, sleep_min, workout_min, 
                    weight, calories_burned, calories_consumed)
    
    if delete_btn and date_exists:
        _handle_delete(df, selected_date)
    
    if clear_btn:
        st.rerun()
    
    # Display recent data table
    _display_recent_records(df)


def _handle_save(df, selected_date, date_exists, steps, sleep_min, workout_min, 
                 weight, calories_burned, calories_consumed):
    """Handle saving a record"""
    new_record = {
        'user_id': '',
        'date': pd.Timestamp(selected_date),
        'steps': steps if steps > 0 else None,
        'sleep_min': sleep_min if sleep_min > 0 else None,
        'workout_duration_min_tot': workout_min if workout_min > 0 else None,
        'weight': weight if weight > 0 else None,
        'calories_burned': calories_burned if calories_burned > 0 else None,
        'calories_consumed': calories_consumed if calories_consumed > 0 else None
    }
    
    try:
        if date_exists:
            # Update existing record
            df.loc[df['date'].dt.date == selected_date, 'steps'] = new_record['steps']
            df.loc[df['date'].dt.date == selected_date, 'sleep_min'] = new_record['sleep_min']
            df.loc[df['date'].dt.date == selected_date, 'workout_duration_min_tot'] = new_record['workout_duration_min_tot']
            df.loc[df['date'].dt.date == selected_date, 'weight'] = new_record['weight']
            df.loc[df['date'].dt.date == selected_date, 'calories_burned'] = new_record['calories_burned']
            df.loc[df['date'].dt.date == selected_date, 'calories_consumed'] = new_record['calories_consumed']
            
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
        
        # Format numeric columns
        numeric_columns = ['steps', 'sleep_min', 'workout_duration_min_tot', 'weight', 
                          'calories_burned', 'calories_consumed']
        for col in numeric_columns:
            if col == 'weight':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "")
            else:
                display_df[col] = display_df[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
        
        # Display the table
        st.dataframe(
            display_df[['date', 'steps', 'sleep_min', 'workout_duration_min_tot', 
                       'weight', 'calories_burned', 'calories_consumed']],
            use_container_width=True,
            column_config={
                'date': st.column_config.TextColumn('Date', width='medium'),
                'steps': st.column_config.TextColumn('Steps', width='small'),
                'sleep_min': st.column_config.TextColumn('Sleep (min)', width='small'), 
                'workout_duration_min_tot': st.column_config.TextColumn('Workout (min)', width='small'),
                'weight': st.column_config.TextColumn('Weight (kg)', width='small'),
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