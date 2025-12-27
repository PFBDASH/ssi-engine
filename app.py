# app.py
import os
import re
import json
import requests
import streamlit as st
import pandas as pd
import engine

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

# -----------------------------
# MEMBERSTACK CONFIG (ENV VARS)
# -----------------------------
# NOTE: Do NOT hardcode secret keys in code.
MEMBERSTACK_API_KEY = os.getenv("MEMBERSTACK_API_KEY")  # set this in Render
# Your Memberstack "price" IDs (you provided these)
PRICE_STARTER = "prc_ssi-starter-x54n0gfn"
PRICE_PRO = "prc_ssi-pro-y54p0hul"
PRICE_BLACK = "prc_ssi-black-d34q0h3p"

# Optional: where to send people to login/signup (your Webflow site)
WEBFLOW_BASE = os.getenv("WEBFLOW_BASE_URL", "https://ssi-auth.webflow.io")
LOGIN_URL = f"{WEBFLOW_BASE}/login"
SIGNUP_URL = f"{WEBFLOW_BASE}/plans"


# -----------------------------
# HELPERS
# -----------------------------
def get_ms_token() -> str | None:
    params = st.query_params  # streamlit >=1.30 style
    token = params.get("ms")
    if isinstance(token, list):
        return token[0] if token else None
    return token


def verify_memberstack_token(token: str) -> dict | None:
    """
    Uses Memberstack Admin REST endpoint:
    POST https://admin.memberstack.com/members/verify-token
    headers: X-API-KEY: <secret>
    body: { "token": "<jwt>" }
    """
    if not MEMBERSTACK_API_KEY:
        return None

    url = "https://admin.memberstack.com/members/verify-token"
    headers = {"X-API-KEY": MEMBERSTACK_API_KEY, "Content-Type": "application/json"}
    try:
        r = requests.post(url, headers=headers, json={"token": token}, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def extract_price_ids(payload: dict) -> set[str]:
    """
    Memberstack response shape can vary. Instead of guessing fields,
    we scan the JSON for any prc_* ids.
    """
    try:
        blob = json.dumps(payload)
    except Exception:
        blob = str(payload)

    matches = re.findall(r"prc_[A-Za-z0-9_-]+", blob)
    return set(matches)


def resolve_tier(price_ids: set[str]) -> str:
    """
    Highest tier wins.
    """
    if PRICE_BLACK in price_ids:
        return "Black"
    if PRICE_PRO in price_ids:
        return "Pro"
    if PRICE_STARTER in price_ids:
        return "Starter"
    return "Free"


def lane_status(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No Data"
    top = df.iloc[0]
    if top.get("status") != "ok":
        return "No Data"
    return "Active" if float(top.get("ssi", 0)) >= 7 else "Stand Down"


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


# -----------------------------
# RUN ENGINE (EVERYONE SEES SSI SCORE)
# -----------------------------
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

ssi = res.get("headline_ssi", 0.0)
banner = res.get("risk_banner", "Unknown")

c1, c2 = st.columns([1, 3])
with c1:
    st.metric("SSI Score", ssi)
with c2:
    st.markdown(f"### {banner}")

st.divider()

# Lane status overview (still OK for everyone)
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

st.subheader("Lane Status (Overview)")
st.dataframe(status_df, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# AUTH + TIER
# -----------------------------
token = get_ms_token()

member_payload = None
tier = "Free"
member_email = None

if token and MEMBERSTACK_API_KEY:
    member_payload = verify_memberstack_token(token)
    if member_payload:
        price_ids = extract_price_ids(member_payload)
        tier = resolve_tier(price_ids)
        # best-effort email extraction
        blob = json.dumps(member_payload)
        m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", blob)
        member_email = m.group(0) if m else None

# If no token, stop after overview
if not token:
    st.info("🔒 Log in to unlock dashboards.")
    colA, colB = st.columns(2)
    with colA:
        st.link_button("Log in", LOGIN_URL)
    with colB:
        st.link_button("Sign up", SIGNUP_URL)
    st.stop()

# If token exists but server key missing, we can’t verify tier securely
if token and not MEMBERSTACK_API_KEY:
    st.error("Server is missing MEMBERSTACK_API_KEY. Add it in Render Environment Variables.")
    st.stop()

# If token invalid
if token and not member_payload:
    st.error("Your login token couldn’t be verified. Try logging in again.")
    st.link_button("Log in", LOGIN_URL)
    st.stop()

# Logged in summary
st.success(f"✅ Logged in{f' as {member_email}' if member_email else ''} — Tier: **{tier}**")

# -----------------------------
# LANE ACCESS RULES
# Starter: choose 1 lane
# Pro: choose up to 2
# Black: all 3
# Free: none
# -----------------------------
all_lanes = ["Crypto", "Forex", "Options"]

if tier == "Free":
    st.info("🔒 You’re on Free. Upgrade to unlock dashboards.")
    st.stop()

max_lanes = 1 if tier == "Starter" else 2 if tier == "Pro" else 3

st.subheader("Dashboards")

selected = st.multiselect(
    f"Select up to {max_lanes} lane(s):",
    options=all_lanes,
    default=all_lanes[:max_lanes],
)

if len(selected) > max_lanes:
    st.warning(f"Too many selected. {tier} allows up to {max_lanes}. Showing first {max_lanes}.")
    selected = selected[:max_lanes]

# Black always gets all 3 (override)
if tier == "Black":
    selected = all_lanes

# Render the chosen lanes
if "Crypto" in selected:
    show_lane("Crypto Lane", crypto_df)

if "Forex" in selected:
    show_lane("Forex Lane", fx_df)

if "Options" in selected:
    show_lane("Options Lane", opt_df)