"""UK road accident severity: triage prototype (Streamlit app).

Run locally:  streamlit run app/streamlit_app.py      (needs models/triage_model.joblib, from `python -m src.deploy_model`)
"""
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.explain import shapley_values  # noqa: E402
from src.features import core_features  # noqa: E402

MODEL_PATH = ROOT / "models" / "triage_model.joblib"
REPO_URL = "https://github.com/ridwanoduola/uk-road-accident-severity"
DEFAULT_LEVEL = "High"
CLASSES = ["slight", "serious", "fatal"]
COLORS = {"slight": "#4C9F70", "serious": "#E0A030", "fatal": "#C0392B"}
INPUT_LABELS = {
    "speed_limit": "Speed limit",
    "light_conditions": "Light conditions",
    "vehicle_type": "Vehicles involved",
    "weather_conditions": "Weather",
    "road_surface_conditions": "Road surface",
}

st.set_page_config(page_title="Road accident severity triage (prototype)", page_icon="🚑", layout="wide")


@st.cache_resource(show_spinner=False)
def load_artifact():
    return joblib.load(MODEL_PATH)


def nice(v):
    s = str(v)
    return s[:1].upper() + s[1:]


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element so it fills its container on ANY Streamlit version:
    newer releases take width="stretch", older ones take use_container_width=True."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except Exception:
        return fn(*args, use_container_width=True, **kwargs)


def pct(x):
    """Percentage text that never claims certainty: shows <1% / >99% at the extremes."""
    return "<1%" if x < 0.005 else (">99%" if x > 0.995 else f"{x:.0%}")


def decide(p, threshold):
    """Flag 'fatal' when P(fatal) reaches the threshold; otherwise the likelier of slight / serious."""
    return 2 if p[2] >= threshold else int(np.argmax(p[:2]))


try:
    A = load_artifact()
except FileNotFoundError:
    st.error("Model file not found. Run `python -m src.deploy_model` first to create `models/triage_model.joblib`.")
    st.stop()
except Exception as exc:  # e.g. library version mismatch when unpickling
    st.error(f"The model could not be loaded ({type(exc).__name__}). It is a pickle, so the installed scikit-learn, pandas "
             "and numpy versions must match the pinned versions in requirements.txt.")
    st.stop()

pipe, lookup, levels, choices = A["pipeline"], A["lookup"], A["levels"], A["choices"]
INPUTS = A["inputs"]


def proba_for(values):
    row = pd.DataFrame([dict(zip(INPUTS, values))])
    row["speed_limit"] = row["speed_limit"].astype(float)
    return pipe.predict_proba(core_features(row))[0]


@st.cache_data(show_spinner=False)
def explain(values):
    x = dict(zip(INPUTS, values))
    fn = lambda df: pipe.predict_proba(core_features(df))[:, 2]
    return shapley_values(fn, x, A["background"], INPUTS)


def default_index(var):
    counts = lookup.groupby(var)[CLASSES].sum().sum(axis=1)
    best = counts.idxmax()
    opts = choices[var]
    return opts.index(int(best) if var == "speed_limit" else best)


# ----------------------------------------------------------------------------------- header
st.title("🚑 Road accident severity triage (prototype)")
st.caption("A machine-learning prototype trained on a 2019 UK road-accident dataset. "
           "It estimates whether an accident is slight, serious or fatal from five facts known at the scene.")
st.warning(
    "**Research prototype, not an operational tool.** The dataset has about "
    f"{A['train_fatal_share']:.0%} fatal accidents, far more than real-world UK figures, so the percentages below describe "
    "**this dataset**, not national risk. Never use this for real emergency decisions.", icon="⚠️")

left, right = st.columns([1, 1.25], gap="large")

# ----------------------------------------------------------------------------------- inputs
with left:
    st.subheader("1. Scene details")
    speed = st.selectbox("Speed limit", choices["speed_limit"], index=default_index("speed_limit"),
                         format_func=lambda v: f"{v} mph", help="Speed limit of the road where the accident happened.")
    light = st.selectbox("Light conditions", choices["light_conditions"], index=default_index("light_conditions"),
                         format_func=nice)
    vehicle = st.selectbox("Vehicles involved", choices["vehicle_type"], index=default_index("vehicle_type"),
                           format_func=nice, help="Category names are used exactly as they appear in the source dataset.")
    weather = st.selectbox("Weather", choices["weather_conditions"], index=default_index("weather_conditions"),
                           format_func=nice)
    surface = st.selectbox("Road surface", choices["road_surface_conditions"],
                           index=default_index("road_surface_conditions"), format_func=nice)

    st.subheader("2. Sensitivity")
    names = list(levels)
    level = st.radio("How cautious should the flag be?", names,
                     index=names.index(DEFAULT_LEVEL) if DEFAULT_LEVEL in names else 0,
                     format_func=lambda n: f"{n}: aims to catch about {levels[n]['target_recall']:.0%} of fatal accidents")
    lv = levels[level]
    st.caption(
        f"On {A['n_test']:,} held-out test accidents, this level caught **{lv['test']['recall_fatal']:.0%}** of the fatal ones "
        f"and flagged **{lv['test']['flagged_share']:.0%}** of all accidents, with "
        f"**{lv['test']['false_alarms_per_fatal_caught']:.1f}** false alarms per fatal accident caught. "
        "Higher sensitivity catches more fatal accidents but raises more false alarms. Which balance is right is a policy "
        "decision, not a modelling one.")

values = (float(speed), light, vehicle, weather, surface)
p = proba_for(values)
pred = decide(p, lv["threshold"])

# ----------------------------------------------------------------------------------- result
with right:
    st.subheader("3. Result")
    if pred == 2:
        st.error(f"**Escalate: flagged as potentially fatal.** The model's estimated share for *fatal* is {pct(p[2])}, "
                 f"which reaches the {level} level's cut-off of {lv['threshold']:.1%}.", icon="🚨")
    else:
        st.info(f"**Not flagged as fatal at the {level} level** (estimated share {pct(p[2])}, cut-off {lv['threshold']:.1%}). "
                f"Most likely category: **{CLASSES[pred]}**. This is not a guarantee: some fatal accidents look like this.",
                icon="ℹ️")
    st.caption("Model-estimated shares within the training dataset:")
    cols = st.columns(3)
    for c, name, prob in zip(cols, CLASSES, p):
        c.metric(nice(name), pct(prob))
        c.progress(float(prob))

    tab_why, tab_similar, tab_model = st.tabs(["Why this estimate?", "Similar past accidents", "Model and limits"])

    with tab_why:
        phi, base, pred_fatal = explain(values)
        st.write(f"The average accident in the dataset has a **{base:.0%}** modelled chance of being fatal. "
                 f"For this scene it is **{pred_fatal:.0%}**. The bars show how much each input moves that figure.")
        d = pd.DataFrame({"Input": [INPUT_LABELS[k] for k in phi],
                          "Change (percentage points)": [v * 100 for v in phi.values()]})
        d["Value"] = [nice(values[i]) if i else f"{int(values[0])} mph" for i in range(len(values))]
        chart = (alt.Chart(d).mark_bar().encode(
            x=alt.X("Change (percentage points):Q", title="Change in estimated chance of 'fatal' (percentage points)"),
            y=alt.Y("Input:N", sort="-x", title=None),
            color=alt.condition(alt.datum["Change (percentage points)"] > 0, alt.value(COLORS["fatal"]), alt.value("#2c7fb8")),
            tooltip=["Input", "Value", alt.Tooltip("Change (percentage points):Q", format="+.1f")]).properties(height=220))
        stretch(st.altair_chart, chart)
        st.caption("These are exact Shapley values of the model's estimate: they describe what the model relies on, "
                   "not what causes accidents.")

    with tab_similar:
        match = lookup[(lookup["speed_limit"] == float(speed)) & (lookup["light_conditions"] == light)
                       & (lookup["vehicle_type"] == vehicle) & (lookup["weather_conditions"] == weather)
                       & (lookup["road_surface_conditions"] == surface)]
        if len(match):
            counts = match[CLASSES].iloc[0]
            n = int(counts.sum())
            st.write(f"In the training data, **{n:,}** accidents had exactly these five conditions:")
            stretch(st.dataframe, pd.DataFrame({"Outcome": [nice(c) for c in CLASSES], "Accidents": counts.astype(int).to_numpy(),
                                                "Share": [f"{v / n:.0%}" for v in counts]}), hide_index=True)
            if n < 20:
                st.caption("Small sample: treat these shares with caution.")
            st.caption("Accidents with identical conditions still had different outcomes, which is why no model can be "
                       "very precise using only these five facts.")
        else:
            st.write("No accident in the training data had exactly this combination, so the model is generalising "
                     "from similar scenes.")

    with tab_model:
        st.markdown(
            f"**Data:** {A['n_train'] + A['n_test']:,} labelled accidents from a 2019 UK dataset "
            f"({A['n_train']:,} used for training, {A['n_test']:,} held out for testing). Category names are used as provided; "
            "their exact definitions were not independently verified.\n\n"
            "**Model:** gradient-boosted trees on five inputs, trained without class weights so its percentages are "
            "calibrated to the dataset. In the full project comparison (12 inputs), a tuned neural network was statistically "
            "indistinguishable from gradient boosting (macro-F1 0.774 vs 0.775 on the test set).")
        tbl = pd.DataFrame([{"Level": n, "Cut-off on P(fatal)": f"{v['threshold']:.1%}",
                             "Fatal accidents caught": f"{v['test']['recall_fatal']:.0%}",
                             "Flags that were fatal": f"{v['test']['precision_fatal']:.0%}",
                             "Share of accidents flagged": f"{v['test']['flagged_share']:.0%}",
                             "False alarms per fatal caught": f"{v['test']['false_alarms_per_fatal_caught']:.2f}"}
                            for n, v in levels.items()])
        stretch(st.dataframe, tbl, hide_index=True)
        st.markdown(
            "**Limitations**\n"
            f"- **Prevalence:** about {A['train_fatal_share']:.0%} of accidents in the data are fatal. At a real-world share of "
            "1% to 2%, the same hit and false-alarm rates would mean roughly 1 in 8 to 1 in 16 fatal flags is correct "
            "(illustration only).\n"
            "- **Limited information:** about 1 in 4 accidents has a different outcome from the typical one for its conditions, "
            "so even a perfect lookup of these five facts would be right only about 77% of the time.\n"
            "- **Weaker areas:** in the test data the model caught only about half of the rare fatal cases in the "
            "'at least one biped' group (31 cases), and it was less accurate in darkness (67%) than in daylight (82%).\n"
            "- **Not validated** for operational use, and not a causal explanation of accidents.")

st.divider()
foot = ("Built with Python, scikit-learn and Streamlit. Contains public sector information licensed under the "
        "Open Government Licence v3.0 (Department for Transport, Road Safety Data).")
if REPO_URL:
    foot += f" [Source code and write-up]({REPO_URL})"
st.caption(foot)
