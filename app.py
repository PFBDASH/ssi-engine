# app.py

import streamlit as st
from engine import (
    run_crypto_scan,
    run_fx_scan,
    run_options_scan,
    compute_ssi,
    recommend_crypto,
    recommend_fx,
    recommend_options,
)

st.set_page_config(page_title="SSI Engine", layout="wide")
st.title("SSI Market Decision Engine")

st.caption("One-click scan. Outputs a clear action when conditions justify it. Thresholds run in the background.")

if st.button("Run Full Scan"):
    crypto = run_crypto_scan()
    fx = run_fx_scan()
    opts = run_options_scan()

    ssi = compute_ssi(crypto)
    st.metric("SSI Score", ssi)

    if ssi >= 7:
        st.success("REGIME: RISK ON")
    elif ssi >= 4:
        st.warning("REGIME: NEUTRAL / CHOP")
    else:
        st.error("REGIME: RISK OFF")

    st.divider()

    # Best rows for each lane
    best_crypto = crypto[0] if crypto else None
    best_fx = fx[0] if fx else None
    best_opt = opts[0] if opts else None

    # Action Box (what you actually want)
    st.subheader("Action Output")
    st.write("**Crypto:** " + recommend_crypto(best_crypto, ssi))
    st.write("**Forex:** " + recommend_fx(best_fx, ssi))
    st.write("**Options:** " + recommend_options(best_opt, ssi))

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Crypto Rankings (Top 5)")
        if crypto:
            st.table(crypto[:5])

    with col2:
        st.subheader("Forex Rankings (Top 5)")
        if fx:
            st.table(fx[:5])

    with col3:
        st.subheader("Options Underlyings (Top 5)")
        if opts:
            st.table(opts[:5])