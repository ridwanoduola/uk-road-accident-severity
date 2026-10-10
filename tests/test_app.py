"""Smoke tests for the Streamlit app (skipped unless Streamlit and models/triage_model.joblib are available)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "triage_model.joblib").exists(),
                                reason="run `python -m src.deploy_model` first")


def _app():
    return AppTest.from_file(APP, default_timeout=90).run()


def test_app_renders_without_errors():
    at = _app()
    assert not at.exception
    assert len(at.selectbox) == 5 and len(at.radio) == 1
    assert len(at.metric) == 3 and len(at.tabs) == 3
    assert len(at.warning) >= 1                       # the research-prototype notice is always shown


def test_extreme_scenes_are_classified_sensibly_at_standard_sensitivity():
    at = _app()
    at.radio[0].set_value("Standard")
    at.selectbox[0].select(70); at.selectbox[1].select("darkness"); at.selectbox[2].select("biped and van")
    at.run()
    assert not at.exception and len(at.error) == 1    # flagged as potentially fatal

    at = _app()
    at.radio[0].set_value("Standard")
    at.selectbox[0].select(20); at.selectbox[1].select("daylight"); at.selectbox[2].select("at least one biped")
    at.selectbox[3].select("fine"); at.selectbox[4].select("dry")
    at.run()
    assert not at.exception and len(at.error) == 0    # not flagged
