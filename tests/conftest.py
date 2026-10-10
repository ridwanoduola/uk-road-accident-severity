import numpy as np
import pandas as pd
import pytest
from src.data import clean


@pytest.fixture(scope="session")
def labeled_df():
    rng = np.random.default_rng(0)
    n = 600
    raw = pd.DataFrame({
        "accident_index": [f"2019{i:08d}" for i in range(n)],
        "speed_limit": rng.choice([20, 30, 40, 60, 70, -1], n, p=[.05, .5, .2, .15, .08, .02]),
        "light_conditions": rng.choice(["daylight", "darkness"], n),
        "weather_conditions": rng.choice(["fine", "other", "data missing or out of range"], n),
        "road_surface_conditions": rng.choice(["dry", "wet or damp", "other"], n),
        "vehicle_type": rng.choice(["only cars", "at least one biped", "at least one van"], n),
        "junction_location": rng.choice(["at or within 20 metres of junction", "data missing or out of range"], n),
        "skidding_and_overturning": rng.choice(["no skidding or overturning", "at least one vehicle skidded or overturned"], n),
        "vehicle_leaving_carriageway": rng.choice(["none leaving carriageway", "at least one vehicle leaving carriageway"], n),
        "hit_object_off_carriageway": rng.choice(["none hit an object", "at least one vehicle hit an object"], n),
        "first_point_of_impact": rng.choice(["at least one vehicle with frontal impact", "other points of impact"], n),
        "sex_of_driver": rng.choice(["all males", "all females", "data missing or out of range"], n),
        "age_of_oldest_driver": np.where(rng.random(n) < .2, np.nan, rng.integers(17, 90, n)).astype(float),
        "accident_severity": rng.choice(["slight", "serious", "fatal"], n, p=[.42, .38, .20]),
    })
    labeled, _, _ = clean(raw)
    return labeled
