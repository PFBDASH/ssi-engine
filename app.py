# app.py
import os
import json
import re
from typing import Any, Dict, List, Optional

import requests
import pandas as pd
import streamlit as st

import engine


# =================================================
# CONFIG
# =================================================
st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

# ---- Env vars (set these in Render) ----
MEMBERSTACK_API_KEY = os.getenv("MEMBERSTACK_API_KEY", "").strip()  # REQUIRED for tier unlocks
WEBFLOW_BASE_URL = os.getenv("WEBFLOW_BASE_URL", "").strip()        # e.g. https://YOUR-SITE.webflow.io or your custom domain

# Your Memberstack "price" IDs (you gave these)
PRICE_STARTER = "prc_ssi-starter-x54n0gfn"
PRICE_PRO     = "prc_ssi-pro-y54p0hul"
PRICE_BLACK   = "prc_ssi-black-d34q0h3p"

# Webflow paths (you specified signup path should be /plans)
LOGIN_PATH  = "/login"
SIGNUP_PATH = "/plans"   # <-- this is your "plans" page

# Lanes
LANES_ALL = ["Crypto", "Forex", "Options"]
CUSTOM_FIELD_KEY = "selected_lanes"   # stored in Memberstack member.customFields.selected_lanes


# =================================================
# UI polish
# =================================================
st.markdown(
    """
<style>
/* Slightly nicer spacing on mobile + cards */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
.card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 12px 14px;
  border-radius: 12px;
}
.smallmuted { color: rgba(255,255,255,0.65); font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# =================================================
# HELPERS
# =================================================
def get_ms_token() -> Optional[str]:
    """
    Memberstack passes a token like: ?ms=xxxxx
    Works with Streamlit >= 1.30 st.query_params
    """
    params = st.query_params
    token = params.get("ms")
    if isinstance(token, list):
        return token[0] if token else None
    return token


def verify_memberstack_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verifies token server-side via Memberstack Admin API
    POST https://admin.memberstack.com/members/verifyToken
    headers: X-API-KEY: <secret>
    body: { "token": "<jwt>" }
    """
    if not MEMBERSTACK_API_KEY:
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
    Best-effort extraction of plan/price IDs from the verifyToken payload.
    Memberstack payload shape can vary by setup, so we scan the JSON blob too.
    """
    if not payload:
        return []

    # 1) Try common structured places
    candidates: List[str] = []

    def walk(obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in ("priceId", "price_id", "planId", "plan_id", "id") and isinstance(v, str):
                    candidates.append(v)
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(payload)

    # 2) Also scan raw json text for prc_... ids
    blob = json.dumps(payload)
    found = re.findall(r"prc_[a-zA-Z0-9\-_]+", blob)
    candidates.extend(found)

    # de-dupe preserve order
    out = []
    seen = set()
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def resolve_tier(price_ids: List[str]) -> str:
    """Returns: Free | Starter | Pro | Black"""
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
    """
    Find member id in common fields or nested member object.
    """
    if not payload:
        return None

    for key in ["id", "memberId", "member_id"]:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    m = payload.get("member")
    if isinstance(m, dict):
        mid = m.get("id")
        if isinstance(mid, str) and mid.strip():
            return mid.strip()

    # fallback: scan blob for something that looks like a member id
    blob = json.dumps(payload)
    # Memberstack member ids often start with "mem_" but not guaranteed; keep conservative
    m2 = re.search(r"(mem_[a-zA-Z0-9]+)", blob)
    return m2.group(1) if m2 else None


def get_member_email(payload: Dict[str, Any]) -> Optional[str]:
    if not payload:
        return None
    for key in ["email", "memberEmail", "member_email"]:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    m = payload.get("member")
    if isinstance(m, dict):
        em = m.get("email")
        if isinstance(em, str) and em.strip():
            return em.strip()

    blob = json.dumps(payload)
    m2 = re.search(r"[\w\.\-]+@[\w\.\-]+\.\w+", blob)
    return m2.group(0) if m2 else None


def get_selected_lanes_from_payload(payload: Dict[str, Any]) -> List[str]:
    """
    Reads member.customFields.selected_lanes if present.
    Accepts:
      - list ["Crypto","Forex"]
      - comma string "Crypto,Forex"
      - json string '["Crypto","Forex"]'
    """
    if not payload:
        return []

    member = payload.get("member") if isinstance(payload.get("member"), dict) else {}
    custom = member.get("customFields") if isinstance(member.get("customFields"), dict) else {}

    raw = custom.get(CUSTOM_FIELD_KEY)

    if raw is None:
        # also allow top-level customFields fallback
        raw2 = payload.get("customFields")
        if isinstance(raw2, dict):
            raw = raw2.get(CUSTOM_FIELD_KEY)

    if raw is None:
        return []

    if isinstance(raw, list):
        lanes = [x for x in raw if isinstance(x, str)]
    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                arr = json.loads(s)
                lanes = [x for x in arr if isinstance(x, str)]
            except Exception:
                lanes = []
        else:
            lanes = [x.strip() for x in s.split(",") if x.strip()]
    else:
        lanes = []

    # sanitize to known lanes
    lanes = [l for l in lanes if l in LANES_ALL]
    # de-dupe preserve order
    out, seen = [], set()
    for l in lanes:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


def update_member_custom_fields(member_id: str, custom_fields: Dict[str, Any]) -> bool:
    """
    PATCH https://admin.memberstack.com/members/{memberId}
    headers: X-API-KEY
    body: { "customFields": {...} }
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

    # recommendation highlight if present
    reco = str(df.iloc[0].get("reco", "") or "").strip()
    if reco:
        st.success(reco)

    cols = [c for c in ["symbol", "last", "trend", "vol", "ssi", "regime"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def build_webflow_url(path: str) -> Optional[str]:
    if not WEBFLOW_BASE_URL:
        return None
    return WEBFLOW_BASE_URL.rstrip("/") + path


# =================================================
# RUN ENGINE (EVERYONE CAN SEE OVERVIEW)
# =================================================
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
    st.caption("This score updates in real time. Only active regimes produce trade opportunities.")
with c2:
    st.markdown(f"### {banner}")

st.divider()

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

st.subheader("Lane Activity")

for _, row in status_df.iterrows():
    lane = row["Lane"]
    status = row["Status"]

    if status == "Active":
        st.success(f"{lane} Lane is ACTIVE")
    elif status == "Stand Down":
        st.warning(f"{lane} Lane is currently in stand-down mode")
    else:
        st.info(f"{lane} Lane status unknown")

st.divider()


# =================================================
# AUTH + TIER RESOLUTION
# =================================================
token = get_ms_token()
login_url = build_webflow_url(LOGIN_PATH)
signup_url = build_webflow_url(SIGNUP_PATH)

member_payload = None
tier = "Free"
member_id = None
member_email = None
price_ids: List[str] = []

if token and MEMBERSTACK_API_KEY:
    member_payload = verify_memberstack_token(token)
    if member_payload:
        price_ids = extract_price_ids(member_payload)
        tier = resolve_tier(price_ids)
        member_id = get_member_id(member_payload)
        member_email = get_member_email(member_payload)

# If no token, we stop after overview and show login/signup CTAs
if not token:
    st.markdown("## The SSI Market Engine")
    st.caption("A regime-aware trading engine that tells you when to stand down — and where to focus when risk is active.")

    st.markdown("### Available Market Lanes")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🪙 Crypto Lane")
        st.write("Detects active crypto regimes and identifies the highest-probability market structures.")
    with c2:
        st.markdown("#### 💱 Forex Lane")
        st.write("Tracks macro + volatility cycles and highlights directional FX opportunity.")
    with c3:
        st.markdown("#### 📈 Options Lane")
        st.write("Identifies high-asymmetry option structures during elevated volatility regimes.")

    st.divider()

    colA, colB = st.columns([2,1])
    with colA:
        if signup_url:
            st.link_button("Start your engine", signup_url, use_container_width=True)
    with colB:
        if login_url:
            st.link_button("Log in", login_url, use_container_width=True)

    st.stop()

# Token exists but server key missing
if token and not MEMBERSTACK_API_KEY:
    st.error("Server is missing MEMBERSTACK_API_KEY (Render env var). Cannot verify access.")
    st.stop()

# Token exists but invalid
if token and not member_payload:
    st.error("Your login token couldn't be verified.")
    if login_url:
        st.link_button("Log in", login_url)
    st.stop()


# =================================================
# LOGGED-IN SUMMARY
# =================================================
left, right = st.columns([3, 2])
with left:
    st.success(f"✅ Logged in{f' as {member_email}' if member_email else ''}")
with right:
    st.markdown(
        f'<div class="card"><b>Tier:</b> {tier}<br><span class="smallmuted">Access determined from your active plan</span></div>',
        unsafe_allow_html=True
    )

st.divider()


# =================================================
# LANE ACCESS + SELECTION
# Rules:
# - Starter: choose 1
# - Pro: choose up to 2
# - Black: all 3 auto
# - Free: none
# =================================================
max_lanes = allowed_lane_count(tier)
selected_lanes = get_selected_lanes_from_payload(member_payload) if member_payload else []

# Free tier: no lanes, show plan CTA
if max_lanes == 0:
    st.info("You’re logged in, but you don’t have an active paid plan.")
    if signup_url:
        st.link_button("Choose a plan", signup_url)
    st.stop()

# Black: auto-select all 3 (write once if needed)
if tier == "Black" and member_id:
    if set(selected_lanes) != set(LANES_ALL):
        ok = update_member_custom_fields(member_id, {CUSTOM_FIELD_KEY: LANES_ALL})
        if ok:
            selected_lanes = LANES_ALL[:]
        else:
            # still allow access locally even if write fails
            selected_lanes = LANES_ALL[:]

# Starter/Pro selection logic (persisted)
def needs_lane_setup() -> bool:
    if tier in ("Starter", "Pro"):
        if len(selected_lanes) == 0:
            return True
        if len(selected_lanes) > max_lanes:
            return True
    return False

needs_setup = needs_lane_setup()

# Allow changing later (Starter/Pro)
if tier in ("Starter", "Pro"):
    with st.expander("⚙️ Manage lanes", expanded=needs_setup):
        st.subheader("Configure Your Market Engine")
st.caption("Choose which intelligence engines to attach to your account.")

        pick = st.multiselect(
            label=f"Select up to {max_lanes}",
            options=LANES_ALL,
            default=selected_lanes[:max_lanes] if selected_lanes else [],
        )

        if len(pick) > max_lanes:
            st.warning(f"Too many selected. Your tier allows {max_lanes}.")
        else:
            if st.button("Save lane selection"):
                if member_id:
                    ok = update_member_custom_fields(member_id, {CUSTOM_FIELD_KEY: pick})
                    if ok:
                        selected_lanes = pick
                        st.success("Saved. Refreshing…")
                        st.rerun()
                    else:
                        st.error("Could not save selection. Try again.")
                else:
                    st.error("Missing member_id in token payload; cannot persist selection.")

# Safety: if downgraded and still has too many saved, trim
if tier in ("Starter", "Pro") and len(selected_lanes) > max_lanes and member_id:
    trimmed = selected_lanes[:max_lanes]
    ok = update_member_custom_fields(member_id, {CUSTOM_FIELD_KEY: trimmed})
    if ok:
        selected_lanes = trimmed


# If still no lanes (shouldn't happen for paid), stop
if not selected_lanes:
    st.warning("Choose your lane(s) to unlock dashboards.")
    st.stop()


# =================================================
# DASHBOARDS — ONLY FOR SELECTED LANES
# =================================================
st.markdown("## Your Market Engine")
st.caption(f"Access Tier: {tier}   |   Active Lanes: {', '.join(selected_lanes)}")
st.divider()

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
st.markdown("### Daily Regime Reminder")
st.caption("The SSI regime can flip overnight. Check in daily to stay on the correct side of risk.")
# Footer controls
cols = st.columns([1, 1, 2])
with cols[0]:
    if signup_url:
        st.link_button("Manage plan / Pricing", signup_url)
with cols[1]:
    if login_url:
        st.link_button("Account / Login", login_url)
with cols[2]:
    st.markdown(
        '<span class="smallmuted">Tip: Bookmark this page after login. Your token in the URL is what unlocks access.</span>',
        unsafe_allow_html=True
    )