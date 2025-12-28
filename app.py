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


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

# Memberstack / Webflow config (ENV VARS)
MEMBERSTACK_API_KEY = os.getenv("MEMBERSTACK_API_KEY", "").strip()

# Webflow base URL (example: https://ssi-auth.webflow.io or your custom domain)
WEBFLOW_BASE_URL = os.getenv("WEBFLOW_BASE_URL", "").strip()

# Admin overrides (ENV VARS)
# - SSI_ADMIN_EMAILS: comma-separated list of emails that should always get Black access
# - SSI_ADMIN_CODE: optional "backdoor" query param admin code for you only (?admin=CODE)
SSI_ADMIN_EMAILS = [e.strip().lower() for e in os.getenv("SSI_ADMIN_EMAILS", "").split(",") if e.strip()]
SSI_ADMIN_CODE = os.getenv("SSI_ADMIN_CODE", "").strip()

# Your Memberstack price IDs (you provided these)
PRICE_STARTER = "prc_ssi-starter-x54n0gfn"
PRICE_PRO = "prc_ssi-pro-y54p0hul"
PRICE_BLACK = "prc_ssi-black-d34q0h3p"

# Lanes
LANES_ALL = ["Crypto", "Forex", "Options"]

# Custom field key in Memberstack where we store lane selection
CUSTOMFIELD_SELECTED_LANES = "selectedLanes"


# =========================================================
# SMALL UI STYLE
# =========================================================
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
      .danger {
        border: 1px solid rgba(255,120,120,0.25);
        background: rgba(255,120,120,0.06);
        border-radius: 14px;
        padding: 0.8rem 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def build_webflow_url(path: str) -> Optional[str]:
    if not WEBFLOW_BASE_URL:
        return None
    return WEBFLOW_BASE_URL.rstrip("/") + path


LOGIN_URL = build_webflow_url("/login")
SIGNUP_URL = build_webflow_url("/plans")  # <-- your clarified signup path


def get_query_param(name: str) -> Optional[str]:
    """
    Streamlit >= 1.30 uses st.query_params; older uses st.experimental_get_query_params()
    """
    # new
    try:
        qp = st.query_params
        v = qp.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        pass

    # old
    try:
        params = st.experimental_get_query_params()
        return params.get(name, [None])[0]
    except Exception:
        return None


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

    # Try structured paths first
    # Memberstack responses can vary; we look in multiple likely places.
    paths_to_check = [
        ("priceIds",),
        ("price_ids",),
        ("prices",),
        ("plans",),
        ("subscriptions",),
        ("member", "plans"),
        ("member", "subscriptions"),
        ("data", "member", "plans"),
        ("data", "member", "subscriptions"),
        ("member", "stripe", "prices"),
        ("data", "member", "stripe", "prices"),
    ]

    for path in paths_to_check:
        node: Any = payload
        ok = True
        for p in path:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                ok = False
                break
        if not ok:
            continue

        if isinstance(node, list):
            for item in node:
                if isinstance(item, str) and item.startswith("prc_"):
                    candidates.append(item)
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, str) and v.startswith("prc_"):
                            candidates.append(v)

    # Regex fallback
    candidates += re.findall(r"prc_[a-zA-Z0-9\-_]+", text)

    # unique preserve order
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


def get_member(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Memberstack verifyToken may return:
    - { member: {...} }
    - { data: { member: {...} } }
    - or other wrappers
    """
    if isinstance(payload.get("member"), dict):
        return payload["member"]
    data = payload.get("data")
    if isinstance(data, dict) and isinstance(data.get("member"), dict):
        return data["member"]
    return None


def get_member_id(payload: Dict[str, Any]) -> Optional[str]:
    member = get_member(payload)
    if isinstance(member, dict):
        v = member.get("id") or member.get("memberId")
        if isinstance(v, str) and v.strip():
            return v.strip()

    # fallback regex
    text = json.dumps(payload)
    m2 = re.search(r'"memberId"\s*:\s*"([^"]+)"', text)
    return m2.group(1) if m2 else None


def get_member_email(payload: Dict[str, Any]) -> Optional[str]:
    member = get_member(payload)
    if isinstance(member, dict):
        v = member.get("email")
        if isinstance(v, str) and v.strip():
            return v.strip()

    text = json.dumps(payload)
    m2 = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return m2.group(0) if m2 else None


def get_selected_lanes_from_payload(payload: Dict[str, Any]) -> List[str]:
    """
    Reads customFields.selectedLanes if present.
    Accepts list or comma-separated string.
    """
    member = get_member(payload)
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

    # filter to known lanes and keep order
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
    """
    US market weekend check (New York time).
    """
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
    if cols:
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


# =========================================================
# RUN SCAN (PUBLIC)
# =========================================================
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


# =========================================================
# PUBLIC OVERVIEW (NO LOGIN REQUIRED)
# =========================================================
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


# =========================================================
# ADMIN BYPASS (OPTIONAL)
# =========================================================
# Allows you (and only you) to get full access without plan.
admin_code = get_query_param("admin")
ADMIN_CODE_OK = bool(SSI_ADMIN_CODE) and (admin_code == SSI_ADMIN_CODE)

# Note: Email-based admin requires login token because we need to know who you are.


# =========================================================
# AUTH GATE (DASHBOARDS)
# =========================================================
token = get_query_param("ms")

# If you use admin code, allow full access without MS token verification.
if ADMIN_CODE_OK:
    tier = "Black"
    max_lanes = 3
    selected_lanes = LANES_ALL[:]
    member_email = "ADMIN"
    st.success("✅ Admin access enabled")
else:
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

    # Email-based admin override (requires login)
    if member_email and member_email.strip().lower() in SSI_ADMIN_EMAILS:
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

    # If paid tier but zero lanes allowed (shouldn't happen)
    if max_lanes == 0:
        st.warning("You are logged in, but you don't have an active plan that unlocks dashboards.")
        if SIGNUP_URL:
            st.link_button("Choose a plan", SIGNUP_URL)
        st.stop()

    # Read saved lanes
    selected_lanes = get_selected_lanes_from_payload(member_payload)

    # Black tier: auto-force all lanes + persist once
    if tier == "Black":
        selected_lanes = LANES_ALL[:]
        if member_id:
            update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: selected_lanes})
    else:
        # Starter/Pro: enforce max_lanes if previously saved too many
        if len(selected_lanes) > max_lanes and member_id:
            trimmed = selected_lanes[:max_lanes]
            ok = update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: trimmed})
            if ok:
                selected_lanes = trimmed

    # If no lanes selected yet (Starter/Pro), force selection
    needs_selection = (tier in ("Starter", "Pro")) and (len(selected_lanes) == 0)

    # =========================================================
    # LANE SELECTION + CHANGE (Starter/Pro)
    # =========================================================
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

    # After selection, if still empty (should not happen), stop
    if not selected_lanes:
        st.warning("Choose your lane(s) to unlock dashboards.")
        st.stop()


# =========================================================
# DASHBOARDS — ONLY FOR SELECTED LANES
# =========================================================
# Weekend hard block for Options dashboard visibility:
if is_weekend_ny() and "Options" in selected_lanes:
    st.markdown(
        '<div class="danger"><b>Options lane is closed on weekends.</b><br>'
        'We will show last-session data, but trading is not available right now.</div>',
        unsafe_allow_html=True,
    )

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