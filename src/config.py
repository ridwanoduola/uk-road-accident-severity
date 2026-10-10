"""Central configuration: paths, seeds, column groups."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "uk_road_accident_2019.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "accidents_clean.csv"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42
TARGET = "accident_severity"
ID_COL = "accident_index"  # corrupted by Excel (scientific notation); dropped, never used as a feature
MISSING_LABEL = "data missing or out of range"

NUMERIC_COLS = ["speed_limit", "age_of_oldest_driver"]
CATEGORICAL_COLS = [
    "light_conditions", "weather_conditions", "road_surface_conditions",
    "vehicle_type", "junction_location", "skidding_and_overturning",
    "vehicle_leaving_carriageway", "hit_object_off_carriageway",
    "first_point_of_impact", "sex_of_driver",
]
# Features only observable AFTER the crash dynamics are known (used for the scene-only ablation)
POST_CRASH_COLS = [
    "skidding_and_overturning", "vehicle_leaving_carriageway",
    "hit_object_off_carriageway", "first_point_of_impact",
]
CLASS_ORDER = ["slight", "serious", "fatal"]
