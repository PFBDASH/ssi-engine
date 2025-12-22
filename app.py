# app.py

import streamlit as st
from engine import (
    run_crypto_scan,
    run_fx_scan,
    run_options_scan,
    compute_ssi,
    recommend_lane,
    recommend_options_contract
)

st.set_page_config(page_title="SSI Engine", layout="wide")
st.title("SSI Market Decision Engine")

st.caption("Opinionated market decision engine. Acts only when conditions justify it.")

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

    best_crypto = crypto[0] if crypto else None
    best_fx = fx[0] if fx else None
    best_opt = opts[0] if opts else None

    st.subheader("Action Output")
    st.write(recommend_lane(best_crypto, ssi, "CRYPTO"))
    st.write(recommend_lane(best_fx, ssi, "FOREX"))
    st.write(recommend_options_contract(best_opt, ssi))

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Crypto Rankings")
        st.table(crypto[:5])

    with col2:
        st.subheader("Forex Rankings")
        st.table(fx[:5])

    with col3:
        st.subheader("Options Underlyings")
        st.table(opts[:5])