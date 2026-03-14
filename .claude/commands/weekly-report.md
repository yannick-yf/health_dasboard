Analyze the last 7 days of data from `data/health_data.csv` and generate a structured weekly report.

Read the CSV, filter to the last 7 complete days, and produce:

## Weekly Report Format

**1. Key Metrics Summary (7-day averages)**
- Average weight (kg) + trend vs prior week
- Average sleep (min → h:mm format)
- Average steps/day
- Average workout duration (min)
- Average calories burned vs consumed
- Average energy balance (surplus/deficit in kcal)

**2. Program Assessment**
- Weight trend: gaining/maintaining/losing? Expected for clean bulk?
- Energy balance: consistent surplus? Average kcal surplus/deficit
- Sleep quality: hitting 480min target?
- Activity load: step consistency, workout frequency

**3. Notable Data Points**
- Best/worst sleep night
- Highest/lowest calorie days
- Any missing data gaps

**4. Recommendation for Next Week**
Based on the trends, suggest 1-2 concrete adjustments to calories, training, or sleep. Reference the clean bulk protocol context from CLAUDE.md.

Use `data/personal_info.json` for BMR/TDEE calculations if needed.
Output should be clean markdown, ready to copy into a note.
