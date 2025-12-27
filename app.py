# app.py
import os
import json
import re
from typing import Dict, Any, List, Optional

import streamlit as st
import pandas as pd
import requests

import engine
from datetime import datetime, timezone

# =================================================
# CONFIG
# =================================================
st.set_page_config(
    page_title="SSI Market Decision Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Required env vars (Render) ---
MEMBERSTACK_API_KEY = os.getenv("MEMBERSTACK_API_KEY", "").strip()  # server-side secret key
WEBFLOW_BASE_URL = os.getenv("WEBFLOW_BASE_URL", "").strip().rstrip("/")  # ex: https://ssi-auth.webflow.io

# --- Your Memberstack "price" IDs (OK to keep in code) ---
PRICE_STARTER = "prc_ssi-starter-x54n0gfn"
PRICE_PRO = "prc_ssi-pro-y54p0hul"
PRICE_BLACK = "prc_ssi-black-d34q0h3p"

# --- Render app base URL (optional; used for small label) ---
APP_BASE_URL = os.getenv("APP_BASE_URL", "").strip().rstrip("/")  # ex: https://ssi-engine.onrender.com

# --- Webflow paths ---
# HARD FIX: Signup now points to /plans (your request)
LOGIN_PATH = os.getenv("LOGIN_PATH", "/login")
SIGNUP_PATH = "/plans"
PRICING_PATH = os.getenv("PRICING_PATH", "/plans")

LOGIN_URL = f"{WEBFLOW_BASE_URL}{LOGIN_PATH}" if WEBFLOW_BASE_URL else ""
SIGNUP_URL = f"{WEBFLOW_BASE_URL}{SIGNUP_PATH}" if WEBFLOW_BASE_URL else ""
PRICING_URL = f"{WEBFLOW_BASE_URL}{PRICING_PATH}" if WEBFLOW_BASE_URL else ""

LANES_ALL = ["Crypto", "Forex", "Options"]
LANE_TO_FIELD = {
    "Crypto": "lane_crypto",
    "Forex": "lane_forex",
    "Options": "lane_options",
}

# =================================================
# LIGHT UI POLISH (CSS)
# =================================================
st.markdown(
    """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.8rem; }
.badge {
  display:inline-block; padding:.25rem .55rem; border-radius:999px;
  font-size:.85rem; border:1px solid rgba(255,255,255,.15);
}
.smallmuted { opacity:.75; font-size:.9rem; }
.card {
  border:1px solid rgba(255,255,255,.10);
  border-radius:16px; padding:16px; background:rgba(255,255,255,.03);
}
</style>
""",
    unsafe_allow_html=True,
)

# =================================================
# HELPERS — QUERY PARAM TOKEN
# =================================================
def get_ms_token() -> Optional[str]:
    params = st.query_params
    token = params.get("ms")
    if isinstance(token, list):
        return token[0] if token else None
    return token

# =================================================
# HELPERS — MEMBERSTACK ADMIN API
# =================================================
def verify_memberstack_token(token: str) -> Optional[Dict[str, Any]]:
    """Validates Memberstack token via Admin API. Returns payload dict if valid, else None."""
    if not MEMBERSTACK_API_KEY:
        return None

    url = "https://admin.memberstack.com/members/verifyToken"
    headers = {"X-API-KEY": MEMBERSTACK_API_KEY, "Content-Type": "application/json"}
    try:
        resp = requests.post(url, headers=headers, json={"token": token}, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, dict) and data.get("data"):
            return data["data"]
        return data if isinstance(data, dict) else None
    except Exception:
        return None

def extract_price_ids(payload: Dict[str, Any]) -> List[str]:
    """Best-effort extraction: find any string like prc_... in the payload."""
    blob = json.dumps(payload)
    return sorted(set(re.findall(r"prc_[A-Za-z0-9\-_]+", blob)))

def resolve_tier(price_ids: List[str]) -> str:
    """Returns: Free | Starter | Pro | Black"""
    if PRICE_BLACK in price_ids:
        return "Black"
    if PRICE_PRO in price_ids:
        return "Pro"
    if PRICE_STARTER in price_ids:
        return "Starter"
    return "Free"

def get_member_id(payload: Dict[str, Any]) -> Optional[str]:
    for key in ["id", "memberId", "member_id"]:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if isinstance(payload.get("member"), dict):
        mid = payload["member"].get("id")
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
    return None

def get_member_email(payload: Dict[str, Any]) -> Optional[str]:
    for key in ["email", "memberEmail", "member_email"]:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    blob = json.dumps(payload)
    m = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", blob)
    return m.group(0) if m else None

def update_member_custom_fields(member_id: str, custom_fields: Dict[str, Any]) -> bool:
    """Updates Memberstack member custom fields."""
    if not MEMBERSTACK_API_KEY:
        return False

    url = f"https://admin.memberstack.com/members/{member_id}"
    headers = {"X-API-KEY": MEMBERSTACK_API_KEY, "Content-Type": "application/json"}
    body = {"customFields": custom_fields}

    try:
        resp = requests.put(url, headers=headers, json=body, timeout=15)
        return resp.status_code in (200, 204)
    except Exception:
        return False

def read_member_lane_prefs(payload: Dict[str, Any]) -> List[str]:
    """Read lane booleans from payload.customFields (best-effort)."""
    cf = payload.get("customFields") or {}
    if not isinstance(cf, dict) and isinstance(payload.get("member"), dict):
        cf = payload["member"].get("customFields") or {}

    lanes = []
    if isinstance(cf, dict):
        for lane, field in LANE_TO_FIELD.items():
            if cf.get(field) is True or str(cf.get(field)).lower() == "true":
                lanes.append(lane)
    return lanes

def allowed_lane_count(tier: str) -> int:
    return {"Starter": 1, "Pro": 2, "Black": 3}.get(tier, 0)

# =================================================
# DATA DISPLAY HELPERS
# =================================================
def lane_status(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No Data"
    top = df.iloc[0]
    if top.get("status") != "ok":
        return "No Data"
    return "Active" if float(top.get("ssi", 0)) >= 7 else "Stand Down"

def show_lane(title: str, df: pd.DataFrame):
    st.markdown(f"### {title}")
    if df is None or df.empty:
        st.warning("No data available.")
        return

    reco = str(df.iloc[0].get("reco", "") or "").strip()
    if reco:
        st.success(reco)

    cols = [c for c in ["symbol", "last", "trend", "vol", "ssi", "regime"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)

# =================================================
# HEADER
# =================================================
st.title("SSI Market Decision Engine")
st.caption("Regime-aware guidance across Crypto, FX, and Options")

token = get_ms_token()

# =================================================
# RUN ENGINE ONCE PER SESSION (so logged-out users still see SSI)
# =================================================
run = st.button("Refresh")

if "result" not in st.session_state:
    st.session_state["result"] = None

if run or st.session_state["result"] is None:
    with st.spinner("Running market scan…"):
        st.session_state["result"] = engine.run_full_scan()
st.session_state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
res = st.session_state["result"]
if not res:
    st.error("Scan failed. Try again.")
    st.stop()

# =================================================
# MARKET REGIME (VISIBLE TO EVERYONE)
# =================================================
ssi = res.get("headline_ssi", 0.0)
banner = res.get("risk_banner", "Unknown")

c1, c2, c3 = st.columns([1.2, 3, 1.2])
with c1:
    st.metric("SSI Score", ssi)
with c2:
    st.markdown(f"### {banner}")
    st.markdown('<span class="smallmuted">Overall market risk + regime summary.</span>', unsafe_allow_html=True)
with c3:
    if APP_BASE_URL:
        st.markdown(f'<span class="badge">App</span> <span class="smallmuted">{APP_BASE_URL}</span>', unsafe_allow_html=True)

st.divider()

# =================================================
# LANE STATUS OVERVIEW (VISIBLE TO EVERYONE)
# =================================================
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

# =================================================
# IF NOT LOGGED IN → SHOW CTA + STOP
# =================================================
if not token:
    st.markdown("## Unlock dashboards")
    st.info("Log in if you already have an account. New? Choose a plan on /plans.")

    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if LOGIN_URL:
            st.link_button("Log in", LOGIN_URL, use_container_width=True)
        else:
            st.warning("Set WEBFLOW_BASE_URL in Render env vars.")
    with colB:
        if SIGNUP_URL:
            st.link_button("Choose a plan", SIGNUP_URL, use_container_width=True)
        else:
            st.warning("Set WEBFLOW_BASE_URL in Render env vars.")
    with colC:
        if PRICING_URL:
            st.link_button("View plans", PRICING_URL, use_container_width=True)

    st.stop()

# =================================================
# LOGGED IN — VERIFY TOKEN, DETERMINE TIER
# =================================================
member_payload = None
member_id = None
member_email = None
tier = "Free"
price_ids = []
selected_lanes = []

if token and MEMBERSTACK_API_KEY:
    member_payload = verify_memberstack_token(token)
    if member_payload:
        price_ids = extract_price_ids(member_payload)
        tier = resolve_tier(price_ids)
        member_id = get_member_id(member_payload)
        member_email = get_member_email(member_payload)
        selected_lanes = read_member_lane_prefs(member_payload)

if token and not MEMBERSTACK_API_KEY:
    st.error("Server is missing MEMBERSTACK_API_KEY. Add it in Render → Service → Environment.")
    st.stop()

if token and not member_payload:
    st.error("Your login token couldn’t be verified. Please log in again.")
    if LOGIN_URL:
        st.link_button("Log in", LOGIN_URL)
    st.stop()

# =================================================
# LOGGED IN SUMMARY
# =================================================
left, right = st.columns([3, 2])
with left:
    st.success(f"✅ Logged in{f' as {member_email}' if member_email else ''}")
with right:
    st.markdown(
        f'<div class="card"><b>Tier:</b> {tier}<br><span class="smallmuted">Lane limit: {allowed_lane_count(tier)}</span></div>',
        unsafe_allow_html=True
    )
## ==============================
# USER CONTROLS
# ==============================
st.divider()
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    refresh = st.button("🔄 Refresh Scan", use_container_width=True)

with c2:
    change = st.button("⚙ Change Lanes", use_container_width=True)

if refresh:
    with st.spinner("Refreshing scan…"):
        st.session_state["result"] = engine.run_full_scan()
    st.rerun()

if change:
    if not member_id:
        st.error("Could not identify your member profile.")
        st.stop()

    # wipe saved lane selections in Memberstack custom fields
    wipe = {LANE_TO_FIELD[l]: False for l in LANES_ALL}
    ok = update_member_custom_fields(member_id, wipe)

    if not ok:
        st.error("Could not reset lane preferences.")
        st.stop()

    st.success("Lane preferences reset — choose new lanes below.")
    st.rerun()
# =================================================
# LANE ACCESS RULES
# Starter: choose 1
# Pro: choose up to 2
# Black: all 3 auto
# Free: none
# =================================================
max_lanes = allowed_lane_count(tier)

if max_lanes == 0:
    st.info("You are logged in, but you don’t have an active plan with lane access.")
    if SIGNUP_URL:
        st.link_button("Choose a plan", SIGNUP_URL)
    st.stop()

# Black auto-select all lanes and write once
if tier == "Black" and member_id:
    if set(selected_lanes) != set(LANES_ALL):
        ok = update_member_custom_fields(member_id, {LANE_TO_FIELD[l]: True for l in LANES_ALL})
        if ok:
            selected_lanes = LANES_ALL[:]

# If no lanes yet OR user has too many lanes (downgrade), force setup
needs_setup = (len(selected_lanes) == 0) or (len(selected_lanes) > max_lanes)

if needs_setup:
    st.subheader("Choose your lanes")
    st.caption("One-time setup. You can change later (we can add a settings page).")

    pick = st.multiselect(
        f"Select up to {max_lanes} lane(s):",
        options=LANES_ALL,
        default=[],
        max_selections=max_lanes,
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        save = st.button("Save lanes", use_container_width=True)
    with col2:
        st.markdown('<span class="smallmuted">Black includes all lanes automatically.</span>', unsafe_allow_html=True)

    if save:
        if not member_id:
            st.error("Could not identify member ID from token payload.")
            st.stop()

        fields = {LANE_TO_FIELD[lane]: (lane in pick) for lane in LANES_ALL}
        ok = update_member_custom_fields(member_id, fields)
        if not ok:
            st.error("Could not save lane preferences. Check MEMBERSTACK_API_KEY and try again.")
            st.stop()

        st.success("Saved. Loading your engine…")
        st.rerun()

    st.stop()

# =================================================
# DASHBOARDS — ONLY FOR SELECTED LANES
# =================================================
st.divider()
st.subheader("Your Engine")

tabs = st.tabs(selected_lanes)
for i, lane in enumerate(selected_lanes):
    with tabs[i]:
        if lane == "Crypto":
            show_lane("Crypto Lane", crypto_df)
        elif lane == "Forex":
            show_lane("Forex Lane", fx_df)
        elif lane == "Options":
            show_lane("Options Lane", opt_df)

st.divider()

cols = st.columns([1, 1, 2])
with cols[0]:
    if SIGNUP_URL:
        st.link_button("Manage plan / Pricing", SIGNUP_URL, use_container_width=True)
with cols[1]:
    if LOGIN_URL:
        st.link_button("Account / Login", LOGIN_URL, use_container_width=True)
with cols[2]:
    st.markdown('<span class="smallmuted">Tip: bookmark after login so you return with your token.</span>', unsafe_allow_html=True)