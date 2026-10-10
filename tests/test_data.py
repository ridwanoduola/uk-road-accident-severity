import pandas as pd
from src import config as cfg
from src.data import clean

BASE = {
    "junction_location": "x", "skidding_and_overturning": "a", "vehicle_leaving_carriageway": "a",
    "hit_object_off_carriageway": "a", "first_point_of_impact": "a", "weather_conditions": "fine",
    "road_surface_conditions": "dry", "sex_of_driver": "all males", "light_conditions": " Darkness",
}


def _toy():
    r = lambda **kw: dict(BASE, **kw)
    rows = [
        r(accident_index="A1", speed_limit=30, vehicle_type="only cars", age_of_oldest_driver=None, accident_severity="Fatal"),
        r(accident_index="A2", speed_limit=-1, vehicle_type="only cars", age_of_oldest_driver=12, accident_severity="slight"),
        r(accident_index="A3", speed_limit=60, vehicle_type="only cars", age_of_oldest_driver=40, accident_severity=""),
        r(accident_index="A4", speed_limit=60, vehicle_type="at least one biped", age_of_oldest_driver=101, accident_severity="serious"),
        r(accident_index="A5", speed_limit=30, vehicle_type="at least one biped", age_of_oldest_driver=11, accident_severity="serious"),
        r(accident_index="A4", speed_limit=60, vehicle_type="at least one biped", age_of_oldest_driver=101, accident_severity="serious"),  # exact dup
    ]
    return pd.DataFrame(rows)


def test_id_dropped_and_text_standardised():
    lab, _, _ = clean(_toy())
    assert cfg.ID_COL not in lab.columns
    assert (lab["light_conditions"] == "darkness").all()


def test_exact_duplicate_record_removed_once():
    lab, unl, log = clean(_toy())
    assert log["exact_duplicate_records_removed"] == 1
    assert log["id_shared_but_not_identical_rows"] == 0
    assert log["n_input"] - 1 == len(lab) + len(unl)


def test_target_variants_recovered_and_blank_set_aside():
    lab, unl, log = clean(_toy())
    assert set(lab[cfg.TARGET]) == {"fatal", "slight", "serious"}
    assert len(unl) == 1
    assert log["target_case_variants_fixed"] == 1


def test_speed_minus_one_becomes_missing_row_kept():
    lab, _, log = clean(_toy())
    assert log["speed_limit_minus1_to_nan"] == 1
    assert lab["speed_missing"].sum() == 1


def test_age_rules():
    lab, _, log = clean(_toy())
    assert log["age_under16_motor_only_set_to_nan"] == 1          # age 12, only cars -> NaN
    assert lab["age_missing"].sum() == 2                          # blank + the implausible one
    biped_child = lab[(lab["vehicle_type"] == "at least one biped") & (lab["age_of_oldest_driver"] == 11)]
    assert len(biped_child) == 1                                  # child cyclist kept
