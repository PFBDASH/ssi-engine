# app.py
import streamlit as st
from engine import (
    run_crypto_scan, run_fx_scan, run_options_scan, compute_ssi,
    crypto_playbook, forex_playbook, options_playbook
)

st.set_page_config(page_title="SSI Engine", layout="wide")
st.title("SSI Market Decision Engine")

with st.expander("Recommendation Gates (controls)", expanded=True):
    min_crypto_score = st.slider("Min score to trigger CRYPTO recommendation", 0.0, 10.0, 6.5, 0.1)
    min_fx_score     = st.slider("Min score to trigger FOREX recommendation", 0.0, 10.0, 6.0, 0.1)
    min_opt_score    = st.slider("Min score to trigger OPTIONS recommendation", 0.0, 10.0, 6.5, 0.1)

run = st.button("Run Full Scan")

if run:
    crypto = run_crypto_scan()
    fx = run_fx_scan()
    opts = run_options_scan()

    ssi = compute_ssi(crypto, fx, opts)
    st.metric("SSI Score", ssi)

    if ssi >= 7:
        st.success("RISK ON — Favor momentum. Directional ideas only if liquidity is good.")
    elif ssi >= 4:
        st.warning("NEUTRAL / CHOP — Selective setups. Smaller size. Prefer mean reversion.")
    else:
        st.error("RISK OFF — Stand down. Preserve capital.")

    st.divider()

    # --- PLAYBOOK OUTPUTS (top of each lane) ---
    st.subheader("Today’s Recommendations (gated)")

    ccol, fcol, ocol = st.columns(3)

    with ccol:
        st.markdown("### Crypto")
        if crypto:
            pb = crypto_playbook(crypto[0], ssi, min_crypto_score)
            if pb["triggered"]:
                st.success(pb["rationale"])
                st.write(f"**Symbol:** {pb['symbol']}  |  **Last:** {pb['last']}")
                st.write(f"**Bias:** {pb['bias']}")
                st.write(f"**Structure:** {pb['structure']}")
            else:
                st.info(pb["rationale"])
        else:
            st.error("No crypto data returned.")

    with fcol:
        st.markdown("### Forex")
        if fx:
            pb = forex_playbook(fx[0], ssi, min_fx_score)
            if pb["triggered"]:
                st.success(pb["rationale"])
                st.write(f"**Pair:** {pb['symbol']}  |  **Last:** {pb['last']}")
                st.write(f"**Bias:** {pb['bias']}")
                st.write(f"**Structure:** {pb['structure']}")
            else:
                st.info(pb["rationale"])
        else:
            st.error("No FX data returned.")

    with ocol:
        st.markdown("### Options")
        if opts:
            pb = options_playbook(opts[0], ssi, min_opt_score)
            if pb["triggered"]:
                st.success(pb["rationale"])
                st.write(f"**Underlying:** {pb['underlying']}  |  **Last:** {pb['last']}")
                st.write(f"**Bias:** {pb['bias']}")
                st.write(f"**Structure:** {pb['structure']}")
                if pb["expiry_days"] is not None:
                    st.write(f"**Expiry heuristic:** {pb['expiry_days']} DTE")
                if pb["target_strike"] is not None:
                    st.write(f"**Target strike zone (approx):** {pb['target_strike']}")
            else:
                st.info(pb["rationale"])
        else:
            st.error("No options-underlying data returned.")

    st.divider()

    # --- TABLES ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Crypto Lane")
        st.caption("Kraken OHLC (hourly).")
        st.table(crypto[:5])

    with col2:
        st.subheader("Forex Lane")
        st.caption("FREE daily OHLC via Stooq.")
        st.table(fx[:5])

    with col3:
        st.subheader("Options Lane")
        st.caption("FREE daily OHLC via Stooq (scores underlyings, not option chains).")
        st.table(opts[:5])

else:
    st.caption("Tap **Run Full Scan** to pull live data and compute SSI.")