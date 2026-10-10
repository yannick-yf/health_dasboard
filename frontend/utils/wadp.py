"""Current Watch-Adaptive Deficit Protocol settings and arithmetic."""

BMR_BASE_KCAL = 2000
CURRENT_DEFICIT_KCAL = 100


def intake_target(move_kcal: float, deficit_kcal: float = CURRENT_DEFICIT_KCAL) -> float:
    return BMR_BASE_KCAL + move_kcal - deficit_kcal


def observed_deficit(move_kcal: float, intake_kcal: float) -> float:
    """Positive values are nominal deficits; negative values are surpluses."""
    return BMR_BASE_KCAL + move_kcal - intake_kcal
