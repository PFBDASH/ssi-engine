# app.py
from __future__ import annotations

import streamlit as st
import pandas as pd

import engine


st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

st.title("SSI Market Decision Engine")
st.caption("Crypto + FX + Options scan → SSI regime + lane recommendations (with contract details where available)")

run = st.button("Run Full Scan")

if "result" not in st.session_state:
    st.session_state["result"] = None

if run:
    with st.spinner("Running scan..."):
        st.session_state["result"] = engine.run_full_scan()

res = st.session_state["result"]

if not res:
    st.info("Tap **Run Full Scan** to generate today’s outputs.")
    st.stop()

# Header summary
col1, col2 = st.columns([1, 3])
with col1:
    st.metric("SSI Score", res["ssi"])
with col2:
    st.markdown(f"### {res['banner']}")
    st.caption(f"As of: {res['asof']}")

st.divider()

# -----------------
# Crypto Lane
# -----------------
st.subheader("Crypto Lane")
crypto_df: pd.DataFrame = res["crypto"]
if crypto_df is None or crypto_df.empty:
    st.warning("Crypto scan returned no data.")
else:
    st.dataframe(crypto_df[["symbol", "direction", "trend", "vol", "score", "rec", "hold"]], use_container_width=True)

    top = crypto_df.iloc[0].to_dict()
    st.markdown("**Crypto Action (if any):**")
    st.write(f"- {top['rec']}  \n- Hold: {top['hold']}")

st.divider()

# -----------------
# Forex Lane
# -----------------
st.subheader("Forex Lane")
fx_df: pd.DataFrame = res["fx"]
if fx_df is None or fx_df.empty:
    st.warning("FX scan returned no data.")
else:
    st.dataframe(fx_df[["symbol", "direction", "trend", "vol", "score", "rec", "hold"]], use_container_width=True)

    top = fx_df.iloc[0].to_dict()
    st.markdown("**FX Action (if any):**")
    st.write(f"- {top['rec']}  \n- Hold: {top['hold']}")

st.divider()

# -----------------
# Options Lane
# -----------------
st.subheader("Options Lane")
opt_df: pd.DataFrame = res["options"]
if opt_df is None or opt_df.empty:
    st.warning("Options scan returned no data.")
else:
    st.dataframe(opt_df[["symbol", "direction", "trend", "vol", "score", "rec", "hold"]], use_container_width=True)

    st.markdown("**Options Setup (only shown for the top candidate if score is high enough):**")
    setup = opt_df.iloc[0].get("options_setup", "")
    if not setup:
        st.info("No options setup triggered (score below threshold or direction neutral).")
    elif isinstance(setup, dict):
        st.json(setup)
    else:
        st.write(setup)