# Health Data ETL Pipeline

This directory contains the ETL (Extract, Transform, Load) pipeline for the Health Dashboard. The ETL process transforms raw health data into enriched data with pre-computed features, improving front-end performance and maintaining consistent calculations.

## Overview

The ETL pipeline takes raw health tracking data and generates:
- **Moving averages** (3, 5, 7 days) for key metrics
- **Derived health metrics** (BMI, BMR, TDEE, energy balance)
- **Convenience conversions** (sleep/workout hours)

This eliminates the need for front-end feature computation and ensures data consistency.

## Architecture

```
┌─────────────────┐
│  Raw CSV Data   │
│ health_data.csv │
└────────┬────────┘
         │ EXTRACT
         ▼
┌─────────────────┐
│  Transform      │
│ - Moving Avg    │
│ - Derived Feat  │
│ - Validation    │
└────────┬────────┘
         │ LOAD
         ▼
┌─────────────────┐
│  Enriched Data  │
│ health_data_    │
│ enriched.csv    │
└─────────────────┘
```

## Features Generated

### Moving Averages (3, 5, 7 days)

All moving averages use `min_periods=1` to ensure values are computed even for the first few days.

**Formula:** `rolling(window=N, min_periods=1).mean()`

Generated for:
- `weight_ma_3`, `weight_ma_5`, `weight_ma_7`
- `steps_ma_3`, `steps_ma_5`, `steps_ma_7`
- `calories_consumed_ma_3`, `calories_consumed_ma_5`, `calories_consumed_ma_7`
- `calories_burned_ma_3`, `calories_burned_ma_5`, `calories_burned_ma_7`
- `sleep_min_ma_3`, `sleep_min_ma_5`, `sleep_min_ma_7`
- `workout_duration_min_tot_ma_3`, `workout_duration_min_tot_ma_5`, `workout_duration_min_tot_ma_7`

### Derived Health Metrics

#### BMI (Body Mass Index)
**Formula:** `BMI = weight(kg) / height(m)²`

Classification:
- Underweight: < 18.5
- Normal: 18.5 - 24.9
- Overweight: 25.0 - 29.9
- Obese: ≥ 30.0

#### BMR (Basal Metabolic Rate)
**Formula:** Mifflin-St Jeor Equation
- Men: `BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age + 5`
- Women: `BMR = 10 × weight(kg) + 6.25 × height(cm) - 5 × age - 161`

#### TDEE (Total Daily Energy Expenditure)
**Formula:** `TDEE = BMR × activity_factor`

Activity factors:
- Sedentary: 1.2 (little/no exercise)
- Lightly active: 1.375 (1-3 days/week)
- Moderately active: 1.55 (3-5 days/week) **[default]**
- Very active: 1.725 (6-7 days/week)
- Extra active: 1.9 (physical job)

#### Energy Balance
**Formula:** `energy_balance = calories_consumed - calories_burned`

- Positive value: Caloric surplus (weight gain)
- Negative value: Caloric deficit (weight loss)
- ~7700 kcal deficit ≈ 1 kg weight loss

### Convenience Conversions

- `sleep_hours` = `sleep_min / 60`
- `workout_hours` = `workout_duration_min_tot / 60`

## Usage

### Command Line

Run the ETL pipeline from the project root:

```bash
# Using Poetry (recommended)
poetry run python etl/transform_health_data.py

# Or with Python directly (if dependencies installed)
python etl/transform_health_data.py
```

### With Custom Paths

```bash
poetry run python etl/transform_health_data.py \
  --input data/health_data.csv \
  --output data/health_data_enriched.csv \
  --personal-info data/personal_info.json
```

### Programmatic Usage

```python
from etl.transform_health_data import run_etl

# Run with default paths
enriched_df = run_etl()

# Or with custom paths
enriched_df = run_etl(
    input_csv='data/health_data.csv',
    output_csv='data/health_data_enriched.csv',
    personal_info_json='data/personal_info.json'
)
```

### Using the ETL Class

```python
from etl.transform_health_data import HealthDataETL

# Initialize ETL processor
etl = HealthDataETL(
    input_csv='data/health_data.csv',
    output_csv='data/health_data_enriched.csv',
    personal_info_json='data/personal_info.json'
)

# Run complete pipeline
enriched_df = etl.run()

# Or run steps individually
raw_df = etl.extract()
transformed_df = etl.transform(raw_df)
etl.load(transformed_df)
```

## Input Files

### health_data.csv

Raw health tracking data with columns:
- `user_id`: User identifier (optional)
- `date`: Date in dd/mm/yyyy format
- `steps`: Daily step count
- `sleep_min`: Sleep duration in minutes
- `workout_duration_min_tot`: Total workout time in minutes
- `weight`: Body weight in kg
- `calories_burned`: Calories burned in kcal
- `calories_consumed`: Calories consumed in kcal

### personal_info.json

Personal information for derived calculations:
```json
{
    "birth_date": "21-01-1992",
    "age": 33,
    "height_cm": 185,
    "sex": "Male",
    "body_fat_prct": 13,
    "activity_factor": 1.55
}
```

Note: If `activity_factor` is not provided, defaults to 1.55 (moderately active).

## Output File

### health_data_enriched.csv

Contains all original columns plus 26 additional features:
- 18 moving average columns (3 windows × 6 metrics)
- 6 derived metrics (BMI, BMR, TDEE, energy_balance, sleep_hours, workout_hours)
- 2 metadata columns (etl_processed_at, etl_version)

**Total columns:** 34

## Formulas Source

All formulas are extracted from the existing front-end codebase:
- **Moving averages:** `frontend/app.py:304-317`
- **BMI:** `frontend/analytics.py:30`
- **BMR:** `frontend/analytics.py:50`
- **TDEE:** `frontend/analytics.py:83`
- **Energy Balance:** `frontend/analytics.py:99`

This ensures consistency between ETL and front-end calculations.

## Logging

The ETL pipeline uses Python's logging module. Logs include:
- Data extraction details (row count, date range)
- Transformation progress (features calculated)
- Loading confirmation (output location)
- Error messages with stack traces

Default log level: `INFO`

## Error Handling

The pipeline handles common issues:
- Missing input files → FileNotFoundError with clear message
- Missing personal_info.json → Uses sensible defaults
- Invalid dates → Date parsing with fallback
- Missing columns → Graceful skipping with warnings

## Scheduling

To run the ETL on a schedule, use:

### Cron (Linux/Mac)

```bash
# Run daily at 2 AM
0 2 * * * cd /path/to/health_dasboard && poetry run python etl/transform_health_data.py >> etl/logs/etl.log 2>&1
```

### Task Scheduler (Windows)

Create a scheduled task to run:
```bash
poetry run python C:\path\to\health_dasboard\etl\transform_health_data.py
```

### Python Scheduler

```python
import schedule
import time
from etl.transform_health_data import run_etl

def job():
    run_etl()
    print("ETL completed")

# Run daily at 2 AM
schedule.every().day.at("02:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Performance

Typical performance metrics:
- **Records:** 53
- **Processing time:** < 1 second
- **Memory usage:** < 50 MB
- **Output file size:** ~15 KB

The pipeline is optimized for datasets up to 10,000 records.

## Testing

Run tests for the ETL module:

```bash
poetry run pytest tests/test_health_tracker.py -v
```

## Troubleshooting

### Issue: "Module not found" error

**Solution:** Install dependencies
```bash
poetry install
```

### Issue: Personal info file not found

**Solution:** The ETL will use defaults, or create the file:
```bash
echo '{"age": 33, "height_cm": 185, "sex": "Male"}' > data/personal_info.json
```

### Issue: Output file not created

**Solution:** Check write permissions on the data/ directory

### Issue: Moving averages seem incorrect

**Solution:** Ensure data is sorted by date (ETL does this automatically)

## Version History

- **v1.0.0** (2025-10-30): Initial release
  - Moving averages (3, 5, 7 days)
  - Derived metrics (BMI, BMR, TDEE, energy balance)
  - Command-line interface
  - Comprehensive logging

## Contributing

When adding new features:
1. Update the `HealthDataETL` class
2. Add feature documentation to this README
3. Update tests
4. Ensure formulas match front-end calculations

## License

This ETL pipeline is part of the Health Dashboard project.
