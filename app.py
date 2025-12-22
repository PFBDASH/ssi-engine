# app.py

import streamlit as st
import pandas as pd

import engine

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="SSI Market Decision Engine",
    layout="wide",
)

# -----------------------------
# HEADER
# -----------------------------
st.title("SSI Market Decision Engine")
st.caption("Regime-based crypto, FX, and options decision framework")

# -----------------------------
# RUN BUTTON
# -----------------------------
run_scan = st.button("Run Full Scan")

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if run_scan:

    # ---- Run scans ----
    crypto_df = engine.run_crypto_scan()
    fx_df = engine.run_fx_scan()
    options_df = engine.run_options_scan()

    # ---- Compute SSI ----
    ssi_score = engine.compute_ssi(
        crypto_df=crypto_df,
        fx_df=fx_df,
        options_df=options_df,
    )

    regime_message = engine.recommend_lane(
        ssi_score=ssi_score,
        crypto_df=crypto_df,
        fx_df=fx_df,
        options_df=options_df,
    )

    # -----------------------------
    # SUMMARY
    # -----------------------------
    st.subheader("Market Regime Summary")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("SSI Score", round(ssi_score, 2))

    with col2:
        st.markdown(
            f"<div style='padding:12px;border-radius:6px;background:#2b2b2b'><strong>{regime_message}</strong></div>",
            unsafe_allow_html=True,
        )

    # -----------------------------
    # CRYPTO LANE
    # -----------------------------
    st.subheader("Crypto Lane")

    if crypto_df is not None and not crypto_df.empty:
        st.dataframe(crypto_df.sort_values("score", ascending=False), use_container_width=True)
        top = crypto_df.sort_values("score", ascending=False).iloc[0]
        if top["score"] >= 7:
            st.success(f"Go LONG {top['symbol']} — target 1–5 day hold")

    # -----------------------------
    # FOREX LANE
    # -----------------------------
    st.subheader("Forex Lane")

    if fx_df is not None and not fx_df.empty:
        st.dataframe(fx_df.sort_values("score", ascending=False), use_container_width=True)
        top = fx_df.sort_values("score", ascending=False).iloc[0]
        direction = "LONG" if top["trend"] > 0 else "SHORT"
        if top["score"] >= 6:
            st.success(f"{direction} {top['symbol']} — swing hold 3–10 days")

    # -----------------------------
    # OPTIONS LANE
    # -----------------------------
    st.subheader("Options Lane")

    if options_df is not None and not options_df.empty:
        st.dataframe(options_df.sort_values("score", ascending=False), use_container_width=True)
        rec = engine.recommend_options_contract(options_df)
        if rec:
            st.success(
                f"{rec['strategy']} | {rec['symbol']} | "
                f"Strike {rec['strike']} | Exp {rec['expiry']} | Bias {rec['bias']}"
            )

else:
    st.info("Press **Run Full Scan** to evaluate the market.")