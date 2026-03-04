Help me add new daily health data entries to `data/health_data.csv`.

1. First, show me the last 3 rows of `data/health_data.csv` so I can see the current state.
2. Ask me for the new data (date, steps, sleep_min, workout_duration_min_tot, weight, calories_burned, calories_consumed).
3. Validate the values are reasonable:
   - steps: 0–50000
   - sleep_min: 180–600
   - workout_duration_min_tot: 0–300
   - weight: 60–90 kg (flag if >2kg change from last entry)
   - calories_burned: 1500–5000
   - calories_consumed: 1000–5000
4. Show me the row that will be appended before writing.
5. Only append after I confirm.

Date format must be YYYY-MM-DD. Missing values should be left empty (not 0).
Never overwrite existing data — append only.
