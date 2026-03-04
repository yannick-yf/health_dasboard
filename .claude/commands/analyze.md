Perform a deep analysis of `data/health_data.csv` focused on the clean bulk program progress.

Read the full dataset and analyze:

**1. Weight Progression**
- Overall trend (linear regression slope in kg/week)
- Phase detection: are there distinct cut/bulk phases visible?
- Current phase assessment vs target (clean bulk = ~0.25-0.5 kg/week gain)

**2. Energy Balance Analysis**
- Average daily surplus/deficit by month
- Correlation between surplus and weight change (lagged by 3-5 days)
- Days where calories_consumed data is missing or suspect

**3. Sleep & Recovery**
- Average sleep trend over time
- Sleep vs next-day performance correlation (steps, workout duration)
- Sleep consistency score (std deviation of sleep_min)

**4. Training Volume**
- Weekly workout minutes trend
- Step count trend (NEAT proxy)
- Total activity load = workout_duration + steps_contribution

**5. Key Correlations**
- Weight change vs energy balance
- Sleep quality vs step count
- Highlight any surprising patterns

**6. Program Timeline**
- Estimate start of current bulk phase from weight/calorie patterns
- Estimated weeks remaining if targeting 0.3 kg/week gain to 76kg

Output as structured markdown with key numbers highlighted.
