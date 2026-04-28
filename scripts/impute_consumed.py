"""
Impute missing `calories_consumed` for 27/03/2026 → 15/04/2026 in
data/health_data_back-up-LAL.csv.

Method:
  1. Train linear regression on days where consumed is logged (Apr 16-27).
     Features: calories_burned, workout_duration_min_tot.
  2. Predict for missing days.
  3. Calibrate so the period-mean prediction satisfies the empirical
     maintenance equation derived from the training window:
        true_TDEE ≈ apple_watch_burned + apple_watch_correction
        avg_consumed_required = avg_burned + correction + Δweight*7700/days
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "health_data_back-up-LAL.csv"
OUT = ROOT / "data" / "health_data_imputed.csv"

KCAL_PER_KG = 7700

df = pd.read_csv(SRC)
df.columns = [c.strip() for c in df.columns]
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"]).reset_index(drop=True)
for c in ["steps", "sleep_min", "workout_duration_min_tot",
          "weight", "calories_burned", "calories_consumed"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

period = df[(df["date"] >= "2026-03-27") & (df["date"] <= "2026-04-27")].copy()

train = period[period["calories_consumed"].notna()].copy()
missing = period[period["calories_consumed"].isna()].copy()

# --- Step 1: empirical correction from training window ---
w_train_start = train.iloc[0]["weight"]
w_train_end = train.iloc[-1]["weight"]
days_train = len(train)
delta_train = w_train_end - w_train_start
avg_consumed_train = train["calories_consumed"].mean()
avg_burned_train = train["calories_burned"].mean()

# empirical TDEE during training window
empirical_tdee_train = avg_consumed_train - (delta_train * KCAL_PER_KG / days_train)
apple_watch_correction = empirical_tdee_train - avg_burned_train

print("=== Training window calibration (Apr 16-27) ===")
print(f"  days={days_train}  weight {w_train_start} → {w_train_end} (Δ={delta_train:+.2f} kg)")
print(f"  avg consumed = {avg_consumed_train:.0f}")
print(f"  avg burned   = {avg_burned_train:.0f}")
print(f"  empirical TDEE = {empirical_tdee_train:.0f}")
print(f"  Apple Watch correction = {apple_watch_correction:+.0f} kcal/day")
print()

# --- Step 2: regression for daily variance ---
X_train = train[["calories_burned", "workout_duration_min_tot"]].values
y_train = train["calories_consumed"].values
# augment with intercept
A = np.column_stack([np.ones(len(X_train)), X_train])
coef, *_ = np.linalg.lstsq(A, y_train, rcond=None)
print(f"=== Regression: consumed = {coef[0]:.0f} + {coef[1]:.2f}*burned + {coef[2]:.2f}*workout_min ===")
y_hat_train = A @ coef
ss_res = ((y_train - y_hat_train) ** 2).sum()
ss_tot = ((y_train - y_train.mean()) ** 2).sum()
r2 = 1 - ss_res / ss_tot
print(f"  R² (in-sample) = {r2:.3f}")
print()

# --- Step 3: predict for missing days ---
X_miss = missing[["calories_burned", "workout_duration_min_tot"]].values
A_miss = np.column_stack([np.ones(len(X_miss)), X_miss])
y_miss_raw = A_miss @ coef

# --- Step 4: calibrate to maintenance equation over missing window ---
w_miss_start = missing.iloc[0]["weight"]
w_miss_end = missing.iloc[-1]["weight"]
days_miss = len(missing)
delta_miss = w_miss_end - w_miss_start
avg_burned_miss = missing["calories_burned"].mean()

# expected avg consumed if Apple Watch correction holds
expected_tdee_miss = avg_burned_miss + apple_watch_correction
expected_avg_consumed = expected_tdee_miss + (delta_miss * KCAL_PER_KG / days_miss)

# scale predictions so their mean matches expected_avg_consumed
shift = expected_avg_consumed - y_miss_raw.mean()
y_miss = y_miss_raw + shift

print(f"=== Missing window calibration (Mar 27 - Apr 15) ===")
print(f"  days={days_miss}  weight {w_miss_start} → {w_miss_end} (Δ={delta_miss:+.2f} kg)")
print(f"  avg burned (Apple Watch) = {avg_burned_miss:.0f}")
print(f"  expected avg consumed = {expected_avg_consumed:.0f}")
print(f"  raw regression mean  = {y_miss_raw.mean():.0f}")
print(f"  shift applied = {shift:+.0f}")
print()

# round to integers
y_miss = np.round(y_miss).astype(int)

# guardrails: clamp to plausible range [1800, 5500]
y_miss = np.clip(y_miss, 1800, 5500)

# write back
period_idx = missing.index
df.loc[period_idx, "calories_consumed"] = y_miss

# preview
preview = df.loc[period_idx, ["date", "steps", "workout_duration_min_tot",
                              "weight", "calories_burned", "calories_consumed"]]
preview = preview.assign(
    date=preview["date"].dt.strftime("%Y-%m-%d")
)
print("=== Imputed values ===")
print(preview.to_string(index=False))

# save
df_out = df.copy()
df_out["date"] = df_out["date"].dt.strftime("%d/%m/%Y")
df_out.to_csv(OUT, index=False)
print(f"\nWrote: {OUT}")
