# Health Tracker Application

A comprehensive health metrics tracking and analysis application built with Python and Streamlit. Track daily health data, visualize trends, and get research-backed insights into your health patterns.

## Features

### 📊 Data Management
- **Daily Record Entry**: Add, edit, and delete health records with validation
- **CSV Import/Export**: Easy data portability
- **Automatic Backups**: Timestamped backups on every data modification
- **Missing Data Handling**: Multiple imputation strategies available

### 📈 Visualizations
- **Interactive Dashboards**: Real-time metrics and KPIs
- **Time Series Analysis**: View trends with 7-day and 30-day moving averages
- **Distribution Analysis**: Histograms and box plots for pattern recognition
- **Correlation Matrix**: Understand relationships between health metrics
- **Data Completeness**: Visual representation of missing data

### 🔬 Deep Dive Analytics
- **BMI Calculation**: Body Mass Index with WHO categories
- **BMR Estimation**: Basal Metabolic Rate using Mifflin-St Jeor equation
- **TDEE Calculation**: Total Daily Energy Expenditure based on activity level
- **Energy Balance**: Track caloric deficit/surplus
- **Sleep Analysis**: Consistency metrics and quality assessment
- **Activity Load**: Weekly workout and step tracking
- **Anomaly Detection**: Statistical identification of unusual patterns
- **Trend Analysis**: Linear regression with significance testing

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager

### Setup

1. Clone or download the repository:
```bash
git clone <repository-url>
cd health-tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Project Structure

```
health-tracker/
├── app.py                 # Main Streamlit application
├── data_utils.py         # Data handling and validation functions
├── analytics.py          # Health metrics calculations
├── visualization.py      # Plotting and visualization helpers
├── requirements.txt      # Python dependencies
├── pyproject.toml       # Project configuration
├── tests/
│   └── test_health_tracker.py  # Unit tests
├── backups/             # Automatic backup directory (created on first backup)
└── health_data.csv      # Default data file (created on first run)
```

## CSV Format

The application expects CSV files with the following columns:

| Column | Type | Description | Valid Range |
|--------|------|-------------|------------|
| user_id | string | User identifier (optional) | - |
| date | string | Date in dd/mm/yyyy format | Valid dates |
| steps | integer | Daily step count | ≥ 0 |
| sleep_min | integer | Sleep duration in minutes | 0-1440 |
| workout_duration_min_tot | integer | Total workout minutes | 0-1440 |
| weight | float | Body weight in kg | 30-250 |
| calories_burned | integer | Calories burned (kcal) |