# app.py
import streamlit as st
import pandas as pd
import engine

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="SSI Market Decision Engine",
    layout="wide"
)

# -------------------------------------------------
# TEMP TIER CONTROL (DEV)
# Replace later with Stripe / auth
# -------------------------------------------------
tier = st.sidebar.selectbox(
    "Access Tier",
    ["Free", "Pro", "Elite"],
    index=0
)

# -------------------------------------------------
# GLOBAL HEADER (ALL USERS)
# -------------------------------------------------
st.title("SSI Market Decision Engine")
st.caption("Crypto + FX + Options → regime-aware trade guidance")

run = st.button("Run Full Scan")

if "result" not in st.session_state:
    st.session_state["result"] = None

if run or st.session_state["result"] is None:
    with st.spinner("Running market scan…"):
        st.session_state["result"] = engine.run_full_scan()

res = st.session_state["result"]
if not res:
    st.error("Scan failed. Try again.")
    st.stop()

# -------------------------------------------------
# MARKET REGIME (ALL USERS)
# -------------------------------------------------
ssi = res.get("headline_ssi", 0.0)
banner = res.get("risk_banner", "Unknown")

c1, c2 = st.columns([1, 3])
with c1:
    st.metric("SSI Score", ssi)
with c2:
    st.markdown(f"### {banner}")

st.divider()

# -------------------------------------------------
# LANE STATUS OVERVIEW (ALL USERS)
# -------------------------------------------------
def lane_status(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No Data"
    top = df.iloc[0]
    if top.get("status") != "ok":
        return "No Data"
    return "Active" if float(top.get("ssi", 0)) >= 7 else "Stand Down"

crypto_df = res.get("crypto")
fx_df = res.get("fx")
opt_df = res.get("options")

status_df = pd.DataFrame(
    [
        {"Lane": "Crypto", "Status": lane_status(crypto_df)},
        {"Lane": "Forex", "Status": lane_status(fx_df)},
        {"Lane": "Options", "Status": lane_status(opt_df)},
    ]
)

st.subheader("Lane Status")
st.dataframe(status_df, use_container_width=True, hide_index=True)

st.divider()

# -------------------------------------------------
# FREE TIER (LOCKED VIEW)
# -------------------------------------------------
if tier == "Free":
    st.subheader("Crypto Lane")
    st.info("🔒 Upgrade to Pro to unlock crypto bias and rankings.")

    st.subheader("Forex Lane")
    st.info("🔒 Upgrade to Pro to unlock FX bias and swing guidance.")

    st.subheader("Options Lane")
    st.info("🔒 Upgrade to Elite to unlock options structures.")

    st.stop()

# -------------------------------------------------
# PRO TIER (CRYPTO + FX)
# -------------------------------------------------
def show_lane(title: str, df: pd.DataFrame):
    st.subheader(title)

    if df is None or df.empty:
        st.warning("No data available.")
        return

    reco = str(df.iloc[0].get("reco", "") or "").strip()
    if reco:
        st.success(reco)

    cols = [c for c in ["symbol", "last", "trend", "vol", "ssi", "regime"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

show_lane("Crypto Lane", crypto_df)
show_lane("Forex Lane", fx_df)

# -------------------------------------------------
# ELITE TIER (OPTIONS)
# -------------------------------------------------
st.subheader("Options Lane")

if tier != "Elite":
    st.info("🔒 Upgrade to Elite to unlock options structures and contract guidance.")
else:
    show_lane("Options Lane", opt_df)