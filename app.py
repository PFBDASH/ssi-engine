# app.py
import os
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import streamlit as st

import engine


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

# Memberstack / Webflow config (ENV VARS)
MEMBERSTACK_API_KEY = os.getenv("MEMBERSTACK_API_KEY", "").strip()

# Webflow base URL (example: https://your-site.webflow.io or your custom domain)
WEBFLOW_BASE_URL = os.getenv("WEBFLOW_BASE_URL", "").strip()

# Admin access list (comma-separated emails)
ADMIN_EMAILS = os.getenv("ADMIN_EMAILS", "").strip()

# Your Memberstack price IDs (you provided these)
PRICE_STARTER = "prc_ssi-starter-x54n0gfn"
PRICE_PRO = "prc_ssi-pro-y54p0hul"
PRICE_BLACK = "prc_ssi-black-d34q0h3p"

# Lanes
LANES_ALL = ["Crypto", "Forex", "Options"]

# Custom field key in Memberstack where we store lane selection
CUSTOMFIELD_SELECTED_LANES = "selectedLanes"


# =========================
# SMALL UI STYLE
# =========================
st.markdown(
    """
    <style>
      .smallmuted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
      .card {
        padding: 0.8rem 1rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
      }
      .pill {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        font-size: 0.85rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# HELPERS
# =========================
def build_webflow_url(path: str) -> Optional[str]:
    if not WEBFLOW_BASE_URL:
        return None
    return WEBFLOW_BASE_URL.rstrip("/") + path


LOGIN_URL = build_webflow_url("/login")
SIGNUP_URL = build_webflow_url("/plans")  # signup/pricing path


def is_admin_email(email: Optional[str]) -> bool:
    """
    Admin allowlist via env var ADMIN_EMAILS="a@b.com,c@d.com"
    """
    if not email:
        return False
    if not ADMIN_EMAILS:
        return False
    admin_set = {e.strip().lower() for e in ADMIN_EMAILS.split(",") if e.strip()}
    return email.strip().lower() in admin_set


def get_ms_token() -> Optional[str]:
    """
    Read ms token from query params: ?ms=<token>
    Supports Streamlit >= 1.30 (st.query_params) and older fallback.
    """
    try:
        qp = st.query_params
        token = qp.get("ms")
        if isinstance(token, list):
            return token[0] if token else None
        return token
    except Exception:
        params = st.experimental_get_query_params()
        return params.get("ms", [None])[0]


def verify_memberstack_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies token with Memberstack Admin API.
    Uses header X-API-KEY: <secret>
    """
    if not token or not MEMBERSTACK_API_KEY:
        return None

    url = "https://admin.memberstack.com/members/verifyToken"
    headers = {
        "X-API-KEY": MEMBERSTACK_API_KEY,
        "Content-Type": "application/json",
    }
    body = {"token": token}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def extract_price_ids(payload: Dict[str, Any]) -> List[str]:
    """
    Best-effort extraction of Memberstack price IDs from verifyToken payload.
    """
    text = json.dumps(payload)
    candidates: List[str] = []

    for key in ["priceIds", "price_ids", "prices", "planConnections", "plans", "subscriptions"]:
        if key in payload and isinstance(payload[key], list):
            for item in payload[key]:
                if isinstance(item, str) and item.startswith("prc_"):
                    candidates.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str) and v.startswith("prc_"):
                            candidates.append(v)

    candidates += re.findall(r"prc_[a-zA-Z0-9\-_]+", text)

    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def resolve_tier(price_ids: List[str]) -> str:
    """
    Returns: Free | Starter | Pro | Black
    """
    if PRICE_BLACK in price_ids:
        return "Black"
    if PRICE_PRO in price_ids:
        return "Pro"
    if PRICE_STARTER in price_ids:
        return "Starter"
    return "Free"


def allowed_lane_count(tier: str) -> int:
    if tier == "Starter":
        return 1
    if tier == "Pro":
        return 2
    if tier == "Black":
        return 3
    return 0


def get_member_id(payload: Dict[str, Any]) -> Optional[str]:
    for key in ["id", "memberId", "member_id"]:
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    m = payload.get("member")
    if isinstance(m, dict):
        v = m.get("id") or m.get("memberId")
        if isinstance(v, str) and v.strip():
            return v.strip()

    text = json.dumps(payload)
    m2 = re.search(r'"memberId"\s*:\s*"([^"]+)"', text)
    return m2.group(1) if m2 else None


def get_member_email(payload: Dict[str, Any]) -> Optional[str]:
    for key in ["email", "memberEmail", "member_email"]:
        v = payload.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()

    m = payload.get("member")
    if isinstance(m, dict):
        v = m.get("email")
        if isinstance(v, str) and v.strip():
            return v.strip()

    text = json.dumps(payload)
    m2 = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return m2.group(0) if m2 else None


def get_selected_lanes_from_payload(payload: Dict[str, Any]) -> List[str]:
    """
    Reads member.customFields.selectedLanes if present.
    Accepts list or comma-separated string.
    """
    member = payload.get("member")
    if not isinstance(member, dict):
        return []

    custom = member.get("customFields")
    if not isinstance(custom, dict):
        return []

    v = custom.get(CUSTOMFIELD_SELECTED_LANES)
    if isinstance(v, list):
        lanes = [x for x in v if isinstance(x, str)]
    elif isinstance(v, str):
        lanes = [x.strip() for x in v.split(",") if x.strip()]
    else:
        lanes = []

    out = []
    for ln in lanes:
        if ln in LANES_ALL and ln not in out:
            out.append(ln)
    return out


def update_member_custom_fields(member_id: str, custom_fields: Dict[str, Any]) -> bool:
    """
    PATCH member custom fields in Memberstack
    """
    if not MEMBERSTACK_API_KEY or not member_id:
        return False

    url = f"https://admin.memberstack.com/members/{member_id}"
    headers = {"X-API-KEY": MEMBERSTACK_API_KEY, "Content-Type": "application/json"}
    body = {"customFields": custom_fields}

    try:
        r = requests.patch(url, headers=headers, json=body, timeout=15)
        return r.status_code in (200, 204)
    except Exception:
        return False


def is_weekend_ny() -> bool:
    now = datetime.now(ZoneInfo("America/New_York"))
    return now.weekday() >= 5  # 5=Sat, 6=Sun


def lane_status_from_df(df: Optional[pd.DataFrame], lane_name: str) -> str:
    """
    Computes "Active" / "Stand Down" / "No Data" plus Options-weekend override.
    """
    if lane_name == "Options" and is_weekend_ny():
        return "Closed (Weekend)"

    if df is None or df.empty:
        return "No Data"
    top = df.iloc[0]
    if str(top.get("status", "")).lower() != "ok":
        return "No Data"
    try:
        ssi_val = float(top.get("ssi", 0))
    except Exception:
        ssi_val = 0.0
    return "Active" if ssi_val >= 7 else "Stand Down"


def show_lane(title: str, df: Optional[pd.DataFrame]):
    st.subheader(title)
    if df is None or df.empty:
        st.warning("No data available.")
        return

    reco = str(df.iloc[0].get("reco", "") or "").strip()
    if reco:
        st.success(reco)

    cols = [c for c in ["symbol", "last", "trend", "vol", "ssi", "regime"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)


# =========================
# RUN SCAN (PUBLIC)
# =========================
st.title("SSI Market Decision Engine")
st.caption("Crypto + FX + Options → regime-aware trade guidance")

run = st.button("Run Full Scan", use_container_width=False)

if "result" not in st.session_state:
    st.session_state["result"] = None

if run or st.session_state["result"] is None:
    with st.spinner("Running market scan…"):
        st.session_state["result"] = engine.run_full_scan()

res = st.session_state["result"]
if not res:
    st.error("Scan failed. Try again.")
    st.stop()

headline_ssi = float(res.get("headline_ssi", 0.0) or 0.0)
banner = str(res.get("risk_banner", "Unknown") or "Unknown")

crypto_df = res.get("crypto")
fx_df = res.get("fx")
opt_df = res.get("options")

# =========================
# PUBLIC OVERVIEW (NO LOGIN REQUIRED)
# =========================
c1, c2 = st.columns([1, 3])
with c1:
    st.metric("SSI Score", f"{headline_ssi:.2f}")
with c2:
    st.markdown(f"### {banner}")

st.caption("This score updates in real time. Only active regimes produce trade opportunities.")
st.divider()

st.markdown("## Lane Activity")
st.caption("High-level status only. Log in to unlock dashboards.")

status_df = pd.DataFrame(
    [
        {"Lane": "Crypto", "Status": lane_status_from_df(crypto_df, "Crypto")},
        {"Lane": "Forex", "Status": lane_status_from_df(fx_df, "Forex")},
        {"Lane": "Options", "Status": lane_status_from_df(opt_df, "Options")},
    ]
)
st.dataframe(status_df, use_container_width=True, hide_index=True)

st.divider()

# =========================
# AUTH GATE (DASHBOARDS)
# =========================
token = get_ms_token()

# Not logged in: show CTA and stop AFTER overview
if not token:
    st.markdown("## Unlock dashboards")
    st.info("Log in if you already have an account. New? Choose a plan on /plans.")

    colA, colB = st.columns(2)
    with colA:
        if LOGIN_URL:
            st.link_button("Log in", LOGIN_URL, use_container_width=True)
        else:
            st.button("Log in (set WEBFLOW_BASE_URL)", disabled=True, use_container_width=True)
    with colB:
        if SIGNUP_URL:
            st.link_button("Choose a plan", SIGNUP_URL, use_container_width=True)
        else:
            st.button("Choose a plan (set WEBFLOW_BASE_URL)", disabled=True, use_container_width=True)

    if not WEBFLOW_BASE_URL:
        st.warning("Set WEBFLOW_BASE_URL in Render env vars to enable login/plan buttons.")
    st.stop()

# If token exists but server key missing -> hard stop (secure)
if token and not MEMBERSTACK_API_KEY:
    st.error("Server is missing MEMBERSTACK_API_KEY. Add it in Render environment variables.")
    st.stop()

member_payload = verify_memberstack_token(token)

if not member_payload:
    st.error("Your login token could not be verified. Please log in again.")
    if LOGIN_URL:
        st.link_button("Log in", LOGIN_URL)
    st.stop()

price_ids = extract_price_ids(member_payload)
tier = resolve_tier(price_ids)
max_lanes = allowed_lane_count(tier)

member_id = get_member_id(member_payload)
member_email = get_member_email(member_payload)

# =========================
# ADMIN OVERRIDE (EMAIL ALLOWLIST)
# =========================
if is_admin_email(member_email):
    tier = "Black"
    max_lanes = 3

# Logged-in summary
left, right = st.columns([3, 2])
with left:
    st.success(f"✅ Logged in{' as ' + member_email if member_email else ''}")
with right:
    st.markdown(
        f'<div class="card"><b>Tier:</b> <span class="pill">{tier}</span><br>'
        f'<span class="smallmuted">Lanes allowed:</span> <b>{max_lanes}</b></div>',
        unsafe_allow_html=True,
    )

# If paid tier but zero lanes allowed
if max_lanes == 0:
    st.warning("You are logged in, but you don't have an active plan that unlocks dashboards.")
    if SIGNUP_URL:
        st.link_button("Choose a plan", SIGNUP_URL)
    st.stop()

# Read saved lanes
selected_lanes = get_selected_lanes_from_payload(member_payload)

# Admin: always all lanes (do not rely on Memberstack write)
if is_admin_email(member_email):
    selected_lanes = LANES_ALL[:]

# Black tier: auto-force all lanes + (optional) persist once
if tier == "Black":
    if set(selected_lanes) != set(LANES_ALL) and member_id:
        ok = update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: LANES_ALL})
        if ok:
            selected_lanes = LANES_ALL[:]
    else:
        selected_lanes = LANES_ALL[:]
else:
    # Starter/Pro: enforce max_lanes if previously saved too many
    if len(selected_lanes) > max_lanes and member_id:
        trimmed = selected_lanes[:max_lanes]
        ok = update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: trimmed})
        if ok:
            selected_lanes = trimmed

# If no lanes selected yet (Starter/Pro), force selection
needs_selection = (tier in ("Starter", "Pro")) and (len(selected_lanes) == 0)

# =========================
# LANE SELECTION + CHANGE (Starter/Pro)
# =========================
if tier in ("Starter", "Pro"):
    with st.expander("⚙️ Manage lanes", expanded=needs_selection):
        st.subheader("Configure Your Market Engine")
        st.caption("Pick which dashboards you want to unlock. You can change this anytime.")

        pick = st.multiselect(
            label=f"Select up to {max_lanes}",
            options=LANES_ALL,
            default=selected_lanes[:max_lanes],
        )

        if len(pick) > max_lanes:
            st.warning(f"Too many selected. Your tier allows {max_lanes}.")
        else:
            if st.button("Save lane selection", use_container_width=True):
                if not member_id:
                    st.error("Missing member_id; cannot save selection.")
                else:
                    ok = update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: pick})
                    if ok:
                        st.success("Saved. Refreshing…")
                        st.rerun()
                    else:
                        st.error("Could not save selection. Try again.")

    if needs_selection:
        st.warning("Choose your lane(s) above to unlock dashboards.")
        st.stop()

# After selection, if still empty, stop
if not selected_lanes:
    st.warning("Choose your lane(s) to unlock dashboards.")
    st.stop()

# =========================
# DASHBOARDS — ONLY FOR SELECTED LANES
# =========================
st.markdown("## Your Market Engine")
st.caption(f"Access Tier: {tier}  |  Active Lanes: {', '.join(selected_lanes)}")
st.divider()

tabs = st.tabs(selected_lanes)

for i, lane in enumerate(selected_lanes):
    with tabs[i]:
        if lane == "Crypto":
            show_lane("Crypto Lane", crypto_df)
        elif lane == "Forex":
            show_lane("Forex Lane", fx_df)
        elif lane == "Options":
            if is_weekend_ny():
                st.info("Options markets are closed on weekends. Signals may reflect last session.")
            show_lane("Options Lane", opt_df)

st.divider()

st.markdown("### Daily Regime Reminder")
st.caption("SSI can flip overnight. Re-check before placing size.")

# Footer controls
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if SIGNUP_URL:
        st.link_button("Manage plan / Pricing", SIGNUP_URL, use_container_width=True)
with col2:
    if LOGIN_URL:
        st.link_button("Account / Login", LOGIN_URL, use_container_width=True)
with col3:
    st.markdown('<span class="smallmuted">Tip: bookmark this page after login.</span>', unsafe_allow_html=True)