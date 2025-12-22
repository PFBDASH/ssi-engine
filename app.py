# app.py

import streamlit as st
from engine import (
    run_crypto_scan,
    run_fx_scan,
    run_options_scan,
    compute_ssi,
    crypto_playbook,
    forex_playbook,
    options_playbook,
)

st.set_page_config(page_title="SSI Engine", layout="wide")
st.title("SSI Market Decision Engine")

with st.expander("Recommendation Gates", expanded=True):
    min_crypto = st.slider("Min Crypto Score", 0.0, 10.0, 6.5, 0.1)
    min_fx = st.slider("Min FX Score", 0.0, 10.0, 6.0, 0.1)
    min_opt = st.slider("Min Options Score", 0.0, 10.0, 6.5, 0.1)

if st.button("Run Full Scan"):
    crypto = run_crypto_scan()
    fx = run_fx_scan()
    opts = run_options_scan()

    ssi = compute_ssi(crypto)
    st.metric("SSI Score", ssi)

    if ssi >= 7:
        st.success("RISK ON")
    elif ssi >= 4:
        st.warning("NEUTRAL / CHOP")
    else:
        st.error("RISK OFF")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Crypto Lane")
        if crypto:
            st.info(crypto_playbook(crypto[0], ssi, min_crypto))
            st.table(crypto[:5])

    with col2:
        st.subheader("Forex Lane")
        if fx:
            st.info(forex_playbook(fx[0], ssi, min_fx))
            st.table(fx[:5])

    with col3:
        st.subheader("Options Lane")
        if opts:
            st.info(options_playbook(opts[0], ssi, min_opt))
            st.table(opts[:5])