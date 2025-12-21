# app.py
import streamlit as st
from engine import run_crypto_scan, compute_ssi

st.set_page_config(page_title="SSI Engine", layout="wide")

st.title("SSI Market Decision Engine")

# ---------- RUN ----------
run = st.button("Run Full Scan")

if run:
    crypto = run_crypto_scan()
    ssi = compute_ssi(crypto)

    st.metric("SSI Score", ssi)

    if ssi >= 7:
        regime = "RISK ON"
        msg = "Risk ON — favor momentum (Crypto / Options lottos if liquid)"
        box = st.success
    elif ssi >= 4:
        regime = "NEUTRAL / CHOP"
        msg = "Chop — favor mean reversion / selective FX"
        box = st.warning
    else:
        regime = "RISK OFF"
        msg = "Risk OFF — stand down / protect capital"
        box = st.error

    box(f"{regime} — {msg}")

    st.divider()

    # ---------- LANES ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Crypto Lane")
        st.caption("Active now. Rankings below.")
        st.table(crypto[:5])

    with col2:
        st.subheader("Forex Lane")
        st.caption("Next: we add FX data + scoring (same framework).")
        st.info("FX scan not wired yet (coming next).")

    with col3:
        st.subheader("Options Lane")
        st.caption("Next: scan underlyings (SPY/QQQ/IWM/NVDA/TSLA) + regime fit.")
        st.info("Options scan not wired yet (coming next).")
else:
    st.caption("Tap **Run Full Scan** to pull live data and compute SSI.")