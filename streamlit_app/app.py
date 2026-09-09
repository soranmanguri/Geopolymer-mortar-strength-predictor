"""
Geopolymer Mortar Strength Predictor -- Streamlit version.

Same tuned XGBoost model (850 trees, 200-trial Optuna search, R2=0.925 on
10-fold CV, 136-mix integrated dataset) as the static HTML/JS calculator at
https://soranmanguri.github.io/Geopolymer-mortar-strength-predictor/ -- a
from-scratch Python reimplementation of the same logic (Sand auto-calc via
volume balance with manual override, live Na2SiO3/NaOH and Alk:Binder
ratios, and the same validation rules: every input must stay within its
trained range, and at least one binder must be nonzero). The actual
calculations live in predictor_logic.py (unit-tested separately -- see
test_predictor_logic.py) so this file only handles widgets/layout.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import streamlit as st

from predictor_logic import (
    FEATURES, FEATURE_KEYS, SG_DEFAULTS, VOID_PCT_DEFAULT, BINDER_KEYS,
    compute_na2sio3_naoh_ratio, compute_alk_binder_ratio, compute_sand_auto,
    compute_violations, build_vector, predict,
)

st.set_page_config(
    page_title="Geopolymer Mortar Strength Predictor",
    page_icon="\U0001F9F1",
    layout="wide",
)

HERE = Path(__file__).parent
MODEL_PATH = HERE / "xgboost_tuned_model.pkl"
TARGET_MIN, TARGET_MAX = 12.87, 82.0

FEATURE_RANGE = {f["key"]: (f["min"], f["max"]) for f in FEATURES}


def range_caption(col, key):
    lo, hi = FEATURE_RANGE[key]
    col.caption(f"Range: {lo:g}–{hi:g}")


DEFAULTS = {
    "Molarity_M": 10.0, "Fly_ash": 343.0, "Slag": 341.0, "Metakaoline": 0.0,
    "Nano_Silica_Kg": 13.0, "NaOH": 108.0, "Na2SiO3": 175.0,
    "Curing_Condition": 25.0, "NS_size_nm": 18.5,
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.markdown(
    """
    <style>
    .big-num {font-size:2.6rem;font-weight:700;color:#8a5c14;}
    h1, h2, h3 {color:#8a5c14 !important;}
    h1 {font-size:2.1rem !important;}
    h3 {font-size:1.2rem !important; text-transform:uppercase; letter-spacing:0.06em;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#f5f2ea; border-radius:10px;
        border:1px solid #ddd3ba !important;
    }
    div[data-testid="stMetricValue"] {color:#201d17;}
    .result-panel {
        background:#eaf3fa; border:1px solid #b8d4e8; border-radius:10px;
        padding:18px 20px;
    }

    /* --- Compact overall layout, bigger text --- */
    .block-container {padding-top:1rem; padding-bottom:1rem; max-width:1180px; margin:0 auto;}
    div[data-testid="stVerticalBlock"] {gap:0.4rem !important;}
    div[data-testid="stHorizontalBlock"] {gap:0.6rem !important;}
    div[data-testid="stVerticalBlockBorderWrapper"] {padding:0.4rem 0.2rem !important;}
    div[data-testid="stMarkdownContainer"] p {font-size:1.08rem !important;}

    /* Bigger labels, bigger input text, narrow (non-stretched) boxes, no +/- steppers */
    [data-testid="stWidgetLabel"] p {font-size:1.22rem !important; font-weight:700;}
    div[data-testid="stNumberInputContainer"] {
        max-width:175px !important;
        border-radius:6px;
        background:#ffffff !important;
        border:1px solid #c9c0a6 !important;
    }
    div[data-testid="stNumberInput"] input {
        font-size:1.4rem !important;
        font-weight:600;
        padding:4px 8px !important;
        height:2.3rem !important;
    }
    div[data-testid="stNumberInput"] button {display:none !important;}
    /* Read-only cells (disabled inputs) should still look like normal cells */
    div[data-testid="stNumberInput"] input:disabled {
        color:#201d17 !important;
        -webkit-text-fill-color:#201d17 !important;
        opacity:1 !important;
    }
    div[data-testid="stNumberInputContainer"]:has(input:disabled) {
        border-color:#c9c0a6 !important;
        opacity:1 !important;
    }
    [data-testid="stWidgetLabel"][disabled] p {
        color:#201d17 !important;
        opacity:1 !important;
    }
    .stCaption, [data-testid="stCaptionContainer"] {font-size:1.02rem !important;}
    div[data-testid="stMetricValue"] {font-size:1.55rem !important;}
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] * {
        font-size:1.0rem !important;
        white-space:normal !important;
        overflow:visible !important;
        text-overflow:unset !important;
    }
    div[data-testid="stButton"] button p {font-size:1.1rem !important;}
    div[data-testid="stAlertContentInfo"] p, div[data-testid="stAlertContentInfo"] {font-size:1.05rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("###### TUNED XGBOOST · OPTUNA-OPTIMIZED, 200 TRIALS")
st.title("\U0001F9F1 Geopolymer Mortar Strength Predictor")
st.write(
    "Predicts 28-day compressive strength of nano-silica-modified fly ash/slag/"
    "metakaolin geopolymer mortar"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Training mixes", "136")
c2.metric("Mix parameters", "12")
c3.metric("Boosted trees", "850")
c4.metric("Observed MPa range", "12.9–82.0")
st.divider()

# ---------------------------------------------------------------- state ---
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("sand_manual", False)
for k, v in SG_DEFAULTS.items():
    st.session_state.setdefault(f"sg_{k}", v)
st.session_state.setdefault("void_pct", VOID_PCT_DEFAULT)
st.session_state.setdefault("last_pred", None)
st.session_state.setdefault("last_violations", [])
st.session_state.setdefault("has_predicted", False)


def reset_to_median():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["Fine_Aggregate"] = None  # recomputed below
    st.session_state.sand_manual = False
    for k, v in SG_DEFAULTS.items():
        st.session_state[f"sg_{k}"] = v
    st.session_state.void_pct = VOID_PCT_DEFAULT
    st.session_state.has_predicted = False


def on_sand_edit():
    st.session_state.sand_manual = True


left, right = st.columns([2, 1], gap="large")

with left:
    binder_box = st.container(border=True)
    with binder_box:
        st.subheader("Binder System")
        st.caption("Precursor solids — at least one of fly ash, slag, or metakaolin.")
        b1, b2, b3 = st.columns(3)
        b1.number_input("Fly Ash (kg/m³)", 0.0, 734.0, key="Fly_ash", step=1.0)
        range_caption(b1, "Fly_ash")
        b2.number_input("GGBFS Slag (kg/m³)", 0.0, 700.0, key="Slag", step=1.0)
        range_caption(b2, "Slag")
        b3.number_input("Metakaolin (kg/m³)", 0.0, 450.0, key="Metakaoline", step=1.0)
        range_caption(b3, "Metakaoline")

    activator_box = st.container(border=True)
    with activator_box:
        st.subheader("Alkaline Activator")
        st.caption("Sodium hydroxide / sodium silicate activation system. The Na₂SiO₃/NaOH ratio is calculated automatically.")
        a1, a2, a3 = st.columns(3)
        a1.number_input("NaOH Molarity (M)", 3.0, 16.0, key="Molarity_M", step=0.5)
        range_caption(a1, "Molarity_M")
        a2.number_input("NaOH Solution (kg/m³)", 64.28, 175.0, key="NaOH", step=0.5)
        range_caption(a2, "NaOH")
        a3.number_input("Na₂SiO₃ Solution (kg/m³)", 127.27, 250.0, key="Na2SiO3", step=0.5)
        range_caption(a3, "Na2SiO3")
        na2sio3_naoh = compute_na2sio3_naoh_ratio(st.session_state.NaOH, st.session_state.Na2SiO3)
        ratio_col = st.columns(3)[0]
        ratio_col.number_input(
            "Na₂SiO₃ / NaOH Ratio", value=na2sio3_naoh, disabled=True, format="%.3f",
        )
        range_caption(ratio_col, "Na2SiO3_NaOH")

    nano_box = st.container(border=True)
    with nano_box:
        st.subheader("Nano-Silica Modification")
        st.caption("Colloidal nano-silica dosage and particle size. Particle size locks to 0 whenever dosage is 0.")
        n1, n2 = st.columns(2)
        n1.number_input("Nano-Silica Dosage (kg/m³)", 0.0, 50.0, key="Nano_Silica_Kg", step=0.5)
        range_caption(n1, "Nano_Silica_Kg")
        ns_size_disabled = st.session_state.Nano_Silica_Kg <= 0
        if ns_size_disabled:
            st.session_state["NS_size_nm"] = 0.0
        n2.number_input(
            "Nano-Silica Particle Size (nm)", 0.0, 30.0, key="NS_size_nm", step=0.5,
            disabled=ns_size_disabled,
            help="No nano-silica in mix — particle size not applicable" if ns_size_disabled else None,
        )
        range_caption(n2, "NS_size_nm")

    physical_box = st.container(border=True)
    with physical_box:
        st.subheader("Physical Mix & Curing")
        alk_binder = compute_alk_binder_ratio(
            st.session_state.NaOH, st.session_state.Na2SiO3,
            st.session_state.Fly_ash, st.session_state.Slag, st.session_state.Metakaoline,
        )

        current_values = {k: st.session_state[k] for k in ["Fly_ash", "Slag", "Metakaoline", "NaOH", "Na2SiO3", "Nano_Silica_Kg"]}
        sg_values = {k: st.session_state[f"sg_{k}"] for k in SG_DEFAULTS}
        sand_auto = compute_sand_auto(current_values, sg_values, st.session_state.void_pct)

        if not st.session_state.sand_manual or st.session_state.get("Fine_Aggregate") is None:
            st.session_state["Fine_Aggregate"] = round(sand_auto, 1)

        p1, p2, p3 = st.columns(3)
        p1.number_input(
            "Sand (kg/m³)", 0.0, 5000.0, key="Fine_Aggregate", step=1.0,
            on_change=on_sand_edit,
            help="Auto-calculated to fill remaining mix volume — edit to override",
        )
        range_caption(p1, "Fine_Aggregate")
        p2.metric("Alkaline / Binder Ratio", f"{alk_binder:.3f}")
        range_caption(p2, "Alk_Binder")
        p3.number_input("Curing Temperature (°C)", 20.0, 70.0, key="Curing_Condition", step=1.0)
        range_caption(p3, "Curing_Condition")

        if st.session_state.sand_manual:
            st.caption("\U0001F4CC Manually entered — click **Reset to median mix** to restore auto-calculation")
        else:
            st.caption(f"Auto-calculated: occupies remaining volume to reach 1000 L (currently {sand_auto:.1f} kg/m³)")

        with st.expander("Mix Volume Check (absolute volume method, specific gravities editable)"):
            sgc = st.columns(4)
            sgc[0].number_input("Fly Ash (SG)", key="sg_Fly_ash")
            sgc[1].number_input("GGBFS Slag (SG)", key="sg_Slag")
            sgc[2].number_input("Metakaolin (SG)", key="sg_Metakaoline")
            sgc[3].number_input("Sand (SG)", key="sg_Fine_Aggregate")
            sgc2 = st.columns(4)
            sgc2[0].number_input("NaOH Solution (SG)", key="sg_NaOH")
            sgc2[1].number_input("Na₂SiO₃ Solution (SG)", key="sg_Na2SiO3")
            sgc2[2].number_input("Nano Silica (SG)", key="sg_Nano_Silica_Kg")
            sgc2[3].number_input("Void Content (%)", key="void_pct")

    bcol1, bcol2 = st.columns([1, 1])
    bcol1.button("Reset to median mix", on_click=reset_to_median, use_container_width=True)
    predict_clicked = bcol2.button("Predict strength", type="primary", use_container_width=True)

with right:
    result_box = st.container(border=True)
    with result_box:
        st.subheader("Predicted strength")

        values = {k: st.session_state[k] for k in ["Molarity_M", "Fly_ash", "Slag", "Metakaoline", "Fine_Aggregate", "Nano_Silica_Kg", "NaOH", "Na2SiO3", "Curing_Condition", "NS_size_nm"]}
        vector = build_vector(values)
        violations = compute_violations(vector)

        if predict_clicked:
            st.session_state.has_predicted = True
            if violations:
                st.session_state.last_pred = None
                st.session_state.last_violations = violations
            else:
                st.session_state.last_pred = predict(model, vector)
                st.session_state.last_violations = []

        if not st.session_state.has_predicted:
            st.markdown('<div class="big-num" style="color:#938a74;">—</div>', unsafe_allow_html=True)
            st.caption("Click Predict strength to run the model")
        elif st.session_state.last_violations:
            st.markdown('<div class="big-num" style="color:#938a74;">—</div>', unsafe_allow_html=True)
            st.error("Out of training range: " + "; ".join(st.session_state.last_violations))
        else:
            pred = st.session_state.last_pred
            st.markdown(f'<div class="big-num">{pred:.1f} <span style="font-size:1.2rem;color:#5e5849;">MPa</span></div>', unsafe_allow_html=True)
            st.caption("28-day compressive strength")
            if TARGET_MIN <= pred <= TARGET_MAX:
                st.success("● Within training range")
            else:
                st.warning("● Outside training range")

        st.divider()
        st.markdown("**Validated model performance**")
        m1, m2 = st.columns(2)
        m1.metric("R² · 10-fold CV", "0.925")
        m2.metric("RMSE · 10-fold CV", "4.35 MPa")
        m1.metric("R² · MC (100×)", "0.898")
        m2.metric("RMSE · MC (100×)", "5.00 MPa")

        st.divider()
        st.markdown("**Mix volume check**")
        if st.session_state.sand_manual:
            sand_vol = st.session_state.Fine_Aggregate / st.session_state.sg_Fine_Aggregate if st.session_state.sg_Fine_Aggregate > 0 else 0
            occupied = sum(current_values[k] / sg_values[k] for k in current_values if sg_values.get(k, 0) > 0)
            total_vol = occupied + (st.session_state.void_pct / 100.0) * 1000.0 + sand_vol
        else:
            total_vol = 1000.0 if sand_auto >= 0 else None
        if total_vol is not None:
            st.metric("Volume", f"{total_vol:.1f} L / 1000 L")

st.divider()
st.caption(
    "Model: XGBoost regressor, hyperparameters selected via 200-trial "
    "Optuna (TPE) search over 10-fold CV R², trained on the full 136-mix "
    "dataset. Source data and full pipeline: "
    "https://github.com/soranmanguri/Geopolymer-mortar-strength-predictor"
)
