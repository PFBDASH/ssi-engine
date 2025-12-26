# app.py
import streamlit as st
import pandas as pd
import requests
import engine

st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

# -------------------------------------------------
# GET MEMBERSTACK SESSION TOKEN
# -------------------------------------------------
params = st.experimental_get_query_params()
token = params.get("ms", [None])[0]

if not token:
    st.error("You must log in to access SSI.")
    st.stop()

# -------------------------------------------------
# VERIFY MEMBER + PLAN
# -------------------------------------------------
try:
    r = requests.get(
        "https://api.memberstack.com/v1/members/me",
        headers={"x-memberstack-session": token},
        timeout=8
    )
    ms = r.json()
except:
    st.error("Authentication failed.")
    st.stop()

plan = (ms.get("plans") or [{}])[0].get("name", "").lower()

# -------------------------------------------------
# LANE ACCESS RULES
# -------------------------------------------------
if "starter" in plan:
    allowed = {"crypto"}
elif "pro" in plan:
    allowed = {"crypto", "fx"}
elif "black" in plan:
    allowed = {"crypto", "fx", "options"}
else:
    allowed = set()

# -------------------------------------------------
# LOAD ENGINE
# -------------------------------------------------
st.title("SSI Market Decision Engine")

with st.spinner("Running scan…"):
    res = engine.run_full_scan()

crypto_df = res.get("crypto")
fx_df = res.get("fx")
opt_df = res.get("options")

def show(title, df):
    if df is None or df.empty:
        st.warning("No data.")
        return
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)

# -------------------------------------------------
# DISPLAY BASE MARKET SSI (EVERYONE)
# -------------------------------------------------
st.metric("Global SSI", res.get("headline_ssi", 0))
st.markdown(f"### {res.get('risk_banner','')}")

st.divider()

# -------------------------------------------------
# LANE GATING
# -------------------------------------------------
if "crypto" in allowed:
    show("Crypto Lane", crypto_df)
else:
    st.info("🔒 Upgrade to unlock Crypto lane.")

if "fx" in allowed:
    show("Forex Lane", fx_df)
else:
    st.info("🔒 Upgrade to unlock Forex lane.")

if "options" in allowed:
    show("Options Lane", opt_df)
else:
    st.info("🔒 Upgrade to unlock Options lane.")