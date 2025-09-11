"""
Data utility functions for Health Tracker application
Handles CSV operations, validation, and data management
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import shutil
import fcntl
import os
import tempfile
from typing import Dict, Tuple, Optional, Any


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse date string in dd/mm/yyyy format
    
    Args:
        date_str: Date string to parse
    
    Returns:
        Parsed datetime object or None if invalid
    """
    if pd.isna(date_str):
        return None
    
    try:
        # Handle both string and date objects
        if isinstance(date_str, (date, datetime)):
            return pd.Timestamp(date_str)
        
        # Try parsing as dd/mm/yyyy
        return pd.to_datetime(date_str, format='%d/%m/%Y', dayfirst=True)
    except:
        try:
            # Fallback to pandas datetime parsing
            return pd.to_datetime(date_str, dayfirst=True)
        except:
            return None


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load health data from CSV file
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with parsed dates and numeric columns
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Parse dates
    if 'date' in df.columns:
        df['date'] = df['date'].apply(parse_date)
        # Remove rows with invalid dates
        df = df[df['date'].notna()].copy()
    
    # Convert numeric columns
    numeric_cols = ['steps', 'sleep_min', 'workout_duration_min_tot', 
                   'weight', 'calories_burned', 'calories_consumed']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    if 'date' in df.columns and not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
    
    return df


def save_data(df: pd.DataFrame, csv_path: str) -> None:
    """
    Save DataFrame to CSV with atomic write operation
    
    Args:
        df: DataFrame to save
        csv_path: Path to save CSV file
    """
    # Ensure directory exists
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Format dates for CSV
    df_save = df.copy()
    if 'date' in df_save.columns:
        df_save['date'] = df_save['date'].dt.strftime('%d/%m/%Y')
    
    # Atomic write: write to temp file then move
    temp_fd, temp_path = tempfile.mkstemp(dir=Path(csv_path).parent, suffix='.tmp')
    
    try:
        # Write to temp file
        df_save.to_csv(temp_path, index=False)
        
        # Atomic rename
        os.replace(temp_path, csv_path)
    finally:
        # Clean up temp file if still exists
        if Path(temp_path).exists():
            os.unlink(temp_path)
        os.close(temp_fd)


def create_backup(csv_path: str) -> str:
    """
    Create timestamped backup of CSV file
    
    Args:
        csv_path: Path to CSV file to backup
    
    Returns:
        Path to backup file
    """
    if not Path(csv_path).exists():
        return ""
    
    # Create backup directory
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"health_data_backup_{timestamp}.csv"
    backup_path = backup_dir / backup_name
    
    # Copy file
    shutil.copy2(csv_path, backup_path)
    
    # Clean old backups (keep only last 10)
    backups = sorted(backup_dir.glob("*.csv"), key=lambda x: x.stat().st_mtime)
    if len(backups) > 10:
        for old_backup in backups[:-10]:
            old_backup.unlink()
    
    return str(backup_path)


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate a health record
    
    Args:
        record: Dictionary containing record data
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check date
    if 'date' not in record or record['date'] is None:
        return False, "Date is required"
    
    # Validate numeric fields
    numeric_fields = {
        'steps': (0, float('inf')),
        'sleep_min': (0, 1440),  # Max 24 hours
        'workout_duration_min_tot': (0, 1440),
        'weight': (30, 250),
        'calories_burned': (0, 10000),
        'calories_consumed': (0, 10000)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in record and record[field] is not None:
            try:
                value = float(record[field])
                if value < min_val or value > max_val:
                    return False, f"{field} must be between {min_val} and {max_val}"
            except (TypeError, ValueError):
                if pd.notna(record[field]):  # Only error if not NaN/None
                    return False, f"Invalid value for {field}"
    
    # Warn about unusual weight values
    if 'weight' in record and record['weight'] is not None:
        weight = float(record['weight'])
        if weight < 40 or weight > 200:
            # This is a warning, not an error
            pass  # Could add warning mechanism here
    
    return True, "Valid"


def append_record(df: pd.DataFrame, record: Dict[str, Any]) -> pd.DataFrame:
    """
    Append a new record to the DataFrame
    
    Args:
        df: Existing DataFrame
        record: New record to append
    
    Returns:
        Updated DataFrame
    """
    # Convert date if needed
    if 'date' in record:
        if isinstance(record['date'], str):
            record['date'] = parse_date(record['date'])
        else:
            record['date'] = pd.Timestamp(record['date'])
    
    # Create new row
    new_row = pd.DataFrame([record])
    
    # Append to existing data
    df_new = pd.concat([df, new_row], ignore_index=True)
    
    # Sort by date
    if 'date' in df_new.columns:
        df_new = df_new.sort_values('date').reset_index(drop=True)
    
    return df_new


def update_record(df: pd.DataFrame, update_date: date, updates: Dict[str, Any]) -> pd.DataFrame:
    """
    Update an existing record by date
    
    Args:
        df: DataFrame to update
        update_date: Date of record to update
        updates: Dictionary of field updates
    
    Returns:
        Updated DataFrame
    """
    df_new = df.copy()
    
    # Find record to update
    mask = df_new['date'] == pd.Timestamp(update_date)
    
    if not mask.any():
        raise ValueError(f"No record found for date {update_date}")
    
    # Update fields
    for field, value in updates.items():
        if field in df_new.columns:
            df_new.loc[mask, field] = value
    
    return df_new


def delete_record(df: pd.DataFrame, delete_date: date) -> pd.DataFrame:
    """
    Delete a record by date
    
    Args:
        df: DataFrame to update
        delete_date: Date of record to delete
    
    Returns:
        Updated DataFrame
    """
    mask = df['date'] != pd.Timestamp(delete_date)
    return df[mask].reset_index(drop=True)


def impute_missing_values(df: pd.DataFrame, method: str = 'none', 
                         custom_values: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Impute missing values in DataFrame
    
    Args:
        df: DataFrame with potential missing values
        method: Imputation method ('none', 'forward_fill', 'median', 'custom')
        custom_values: Dictionary of custom values for each column
    
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    numeric_cols = ['steps', 'sleep_min', 'workout_duration_min_tot', 
                   'weight', 'calories_burned', 'calories_consumed']
    
    if method == 'none':
        return df_imputed
    
    elif method == 'forward_fill':
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(method='ffill')
    
    elif method == 'median':
        for col in numeric_cols:
            if col in df_imputed.columns:
                median_val = df_imputed[col].median()
                df_imputed[col] = df_imputed[col].fillna(median_val)
    
    elif method == 'custom' and custom_values:
        for col, value in custom_values.items():
            if col in df_imputed.columns:
                df_imputed[col] = df_imputed[col].fillna(value)
    
    return df_imputed


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for the data
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {
            'total_records': 0,
            'date_range': None,
            'completeness': {}
        }
    
    numeric_cols = ['steps', 'sleep_min', 'workout_duration_min_tot', 
                   'weight', 'calories_burned', 'calories_consumed']
    
    summary = {
        'total_records': len(df),
        'date_range': (df['date'].min(), df['date'].max()),
        'completeness': {}
    }
    
    # Calculate completeness for each column
    for col in numeric_cols:
        if col in df.columns:
            completeness = df[col].notna().sum() / len(df) * 100
            summary['completeness'][col] = completeness
    
    return summary