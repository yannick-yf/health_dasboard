"""
Body fat percentage estimators using waist + weight + personal info.

Three formulas implemented (all male-validated; female formulas differ):

1. RFM (Relative Fat Mass, Woolcott & Bergman 2018, Sci Rep)
   - Inputs: height (m), waist (m), sex
   - Validated against DEXA on 12,000+ adults; widely used as a DEXA-proxy.
   - Formula (men): 64 − 20 × (height / waist)

2. Deurenberg BMI-based (Deurenberg et al. 1991, Br J Nutr)
   - Inputs: BMI, age, sex
   - Formula: 1.20 × BMI + 0.23 × age − 10.8 × sex − 5.4
     (sex = 1 for male, 0 for female)
   - Tends to under-estimate BF for muscular trained males.

3. YMCA (waist + weight, Wallace & Ray 1991)
   - Inputs: waist (inches), weight (lbs), sex
   - Formula (men): (−98.42 + 4.15 × waist_in − 0.082 × weight_lbs) / weight_lbs × 100
   - Quick population estimator, accuracy ±4% vs DEXA on average.

Returns single estimates and a consensus (mean across the three).
"""

from typing import Optional, Dict


def rfm_male(height_cm: float, waist_cm: float) -> float:
    """Relative Fat Mass estimate for males. Returns body fat %."""
    if height_cm <= 0 or waist_cm <= 0:
        return float("nan")
    return 64.0 - 20.0 * (height_cm / waist_cm)


def deurenberg_male(weight_kg: float, height_cm: float, age_yrs: int) -> float:
    """Deurenberg BMI-based BF % estimate for males. Returns body fat %."""
    if weight_kg <= 0 or height_cm <= 0 or age_yrs <= 0:
        return float("nan")
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    return 1.20 * bmi + 0.23 * age_yrs - 10.8 * 1 - 5.4


def ymca_male(waist_cm: float, weight_kg: float) -> float:
    """YMCA waist+weight BF % estimate for males. Returns body fat %."""
    if waist_cm <= 0 or weight_kg <= 0:
        return float("nan")
    waist_in = waist_cm / 2.54
    weight_lbs = weight_kg * 2.20462
    return (-98.42 + 4.15 * waist_in - 0.082 * weight_lbs) / weight_lbs * 100.0


def estimate_body_fat(
    waist_cm: Optional[float],
    weight_kg: Optional[float],
    height_cm: float,
    age_yrs: int,
    sex: str = "Male",
) -> Dict[str, Optional[float]]:
    """
    Run all available BF estimators given the inputs.

    Returns dict with each estimator's result + a consensus (mean of available).
    Estimators return None if their required inputs are missing.

    Currently male-only — female formulas would be added if needed.
    """
    if sex.lower() not in ("male", "m"):
        return {
            "rfm": None, "deurenberg": None, "ymca": None, "consensus": None,
            "lean_mass_kg": None, "fat_mass_kg": None,
        }

    rfm = (
        rfm_male(height_cm, waist_cm)
        if waist_cm is not None and waist_cm > 0
        else None
    )
    deurenberg = (
        deurenberg_male(weight_kg, height_cm, age_yrs)
        if weight_kg is not None and weight_kg > 0
        else None
    )
    ymca = (
        ymca_male(waist_cm, weight_kg)
        if waist_cm is not None and waist_cm > 0
        and weight_kg is not None and weight_kg > 0
        else None
    )

    available = [v for v in (rfm, deurenberg, ymca) if v is not None]
    consensus = sum(available) / len(available) if available else None

    lean_mass = None
    fat_mass = None
    if consensus is not None and weight_kg is not None and weight_kg > 0:
        fat_mass = weight_kg * consensus / 100.0
        lean_mass = weight_kg - fat_mass

    return {
        "rfm": round(rfm, 1) if rfm is not None else None,
        "deurenberg": round(deurenberg, 1) if deurenberg is not None else None,
        "ymca": round(ymca, 1) if ymca is not None else None,
        "consensus": round(consensus, 1) if consensus is not None else None,
        "lean_mass_kg": round(lean_mass, 2) if lean_mass is not None else None,
        "fat_mass_kg": round(fat_mass, 2) if fat_mass is not None else None,
    }
