import streamlit as st
import pandas as pd

from engine import (
    run_crypto_scan,
    run_fx_scan,
    run_options_scan,
    compute_ssi,
    recommend_lane,
    recommend_options_contract,
)

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
    crypto_df = run_crypto_scan()
    fx_df = run_fx_scan()
    options_df = run_options_scan()

    # ---- Compute overall SSI ----
    ssi_score = compute_ssi(
        crypto_df=crypto_df,
        fx_df=fx_df,
        options_df=options_df,
    )

    # ---- Top-level recommendation ----
    regime_message = recommend_lane(
        ssi_score=ssi_score,
        crypto_df=crypto_df,
        fx_df=fx_df,
        options_df=options_df,
    )

    # -----------------------------
    # SUMMARY PANEL
    # -----------------------------
    st.subheader("Market Regime Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("SSI Score", round(ssi_score, 2))

    with col2:
        st.markdown(
            f"""
            <div style="padding:12px;border-radius:6px;background-color:#2b2b2b">
            <strong>{regime_message}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------
    # CRYPTO LANE
    # -----------------------------
    st.subheader("Crypto Lane")

    if crypto_df is not None and not crypto_df.empty:
        st.dataframe(
            crypto_df.sort_values("score", ascending=False),
            use_container_width=True,
        )

        top_crypto = crypto_df.sort_values("score", ascending=False).iloc[0]

        if top_crypto["score"] >= 7:
            st.success(
                f"Go LONG {top_crypto['symbol']} — trend aligned. "
                f"Target hold: 1–5 days."
            )

    else:
        st.warning("Crypto scan returned no data.")

    # -----------------------------
    # FOREX LANE
    # -----------------------------
    st.subheader("Forex Lane")

    if fx_df is not None and not fx_df.empty:
        st.dataframe(
            fx_df.sort_values("score", ascending=False),
            use_container_width=True,
        )

        top_fx = fx_df.sort_values("score", ascending=False).iloc[0]

        if top_fx["score"] >= 6:
            direction = "LONG" if top_fx["trend"] > 0 else "SHORT"
            st.success(
                f"{direction} {top_fx['symbol']} — swing bias. "
                f"Target hold: 3–10 days."
            )

    else:
        st.warning("FX scan returned no data.")

    # -----------------------------
    # OPTIONS LANE
    # -----------------------------
    st.subheader("Options Lane")

    if options_df is not None and not options_df.empty:
        st.dataframe(
            options_df.sort_values("score", ascending=False),
            use_container_width=True,
        )

        best_contract = recommend_options_contract(options_df)

        if best_contract is not None:
            st.success(
                f"Options Play: {best_contract['strategy']} on "
                f"{best_contract['symbol']} | "
                f"Strike: {best_contract['strike']} | "
                f"Expiry: {best_contract['expiry']} | "
                f"Bias: {best_contract['bias']}"
            )
    else:
        st.warning("Options scan returned no data.")

else:
    st.info("Press **Run Full Scan** to evaluate current market conditions.")