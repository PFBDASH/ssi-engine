# app.py
import streamlit as st
from engine import run_crypto_scan, run_fx_scan, run_options_scan, compute_ssi

st.set_page_config(page_title="SSI Engine", layout="wide")
st.title("SSI Market Decision Engine")

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

        # Simple playbook cue (structure suggestion, not strike/expiry)
        top = opts[0]["symbol"] if opts else None
        if top:
            if ssi >= 7:
                st.success(f"Playbook cue: Momentum environment. If trading options, start by watching **{top}** for trend-follow setups.")
            elif ssi >= 4:
                st.warning(f"Playbook cue: Chop. If trading options, consider range/defined-risk structures on **{top}** (not lottos).")
            else:
                st.error("Playbook cue: Risk-off. Options lottos are lowest-quality here.")
else:
    st.caption("Tap **Run Full Scan** to pull live data and compute SSI.")