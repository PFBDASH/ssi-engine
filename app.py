# app.py
import streamlit as st
from engine import run_crypto_scan, compute_ssi

st.set_page_config(page_title="SSI Engine", layout="wide")

st.title("SSI Market Decision Engine")

if st.button("Run Full Scan"):
    crypto = run_crypto_scan()
    ssi = compute_ssi(crypto)

    st.metric("SSI Score", ssi)

    if ssi >= 7:
        st.success("Risk ON — Crypto Momentum / Options Lottos")
    elif ssi >= 4:
        st.warning("Chop — Forex or Mean Reversion")
    else:
        st.error("Risk OFF — Stand Down")

    st.subheader("Crypto Rankings")
    st.table(crypto[:5])