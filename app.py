import os
import json
import re
import base64
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
WEBFLOW_BASE_URL = os.getenv("WEBFLOW_BASE_URL", "").strip()

WEBFLOW_LOGIN_PATH = os.getenv("WEBFLOW_LOGIN_PATH", "/sign-in").strip() or "/sign-in"
WEBFLOW_PLANS_PATH = os.getenv("WEBFLOW_PLANS_PATH", "/plans").strip() or "/plans"

# Admin overrides (ENV VARS)
SSI_ADMIN_EMAILS = [e.strip().lower() for e in os.getenv("SSI_ADMIN_EMAILS", "").split(",") if e.strip()]
SSI_ADMIN_MEMBER_IDS = [m.strip() for m in os.getenv("SSI_ADMIN_MEMBER_IDS", "").split(",") if m.strip()]
SSI_ADMIN_CODE = os.getenv("SSI_ADMIN_CODE", "").strip()

# Your Memberstack price IDs
PRICE_STARTER = "prc_ssi-starter-x54n0gfn"
PRICE_PRO = "prc_ssi-pro-y54p0hul"
PRICE_BLACK = "prc_ssi-black-d34q0h3p"

# Lanes (now 4)
LANES_ALL = ["Crypto", "Forex", "Options", "Long Cycle"]  # Lane 4 = US equities only
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
  .brandbar {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.8rem;
  }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def ny_now() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))

def ny_date_key() -> str:
    return ny_now().strftime("%Y-%m-%d")

def is_weekend_ny() -> bool:
    return ny_now().weekday() >= 5

def build_webflow_url(path: str) -> Optional[str]:
    if not WEBFLOW_BASE_URL:
        return None
    if not path.startswith("/"):
        path = "/" + path
    return WEBFLOW_BASE_URL.rstrip("/") + path

LOGIN_URL = build_webflow_url(WEBFLOW_LOGIN_PATH)
PLANS_URL = build_webflow_url(WEBFLOW_PLANS_PATH)

def get_query_param(name: str) -> Optional[str]:
    # new streamlit
    try:
        qp = st.query_params
        v = qp.get(name)
        if isinstance(v, list):
            return v[0] if v else None
        return v
    except Exception:
        pass
    # older streamlit
    try:
        params = st.experimental_get_query_params()
        return params.get(name, [None])[0]
    except Exception:
        return None

def _jwt_member_id(token: str) -> Optional[str]:
    """Decode JWT payload without verifying signature, to extract member id for lookup."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return None
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        data = json.loads(payload_json)
        mid = data.get("id")
        return mid if isinstance(mid, str) and mid.strip() else None
    except Exception:
        return None

def verify_memberstack_token(token: str) -> Optional[Dict[str, Any]]:
    if not token or not MEMBERSTACK_API_KEY:
        return None

    headers = {
        "X-API-KEY": MEMBERSTACK_API_KEY,
        "Content-Type": "application/json",
    }

    # Verify token
    try:
        r = requests.post(
            "https://admin.memberstack.com/members/verify-token",
            headers=headers,
            json={"token": token},
            timeout=15,
        )
        if r.status_code != 200:
            return None
        verify_payload = r.json()
    except Exception:
        return None

    # Fetch full profile by member id from JWT
    member_id = _jwt_member_id(token)
    if not member_id:
        return verify_payload

    try:
        r2 = requests.get(
            f"https://admin.memberstack.com/members/{member_id}",
            headers=headers,
            timeout=15,
        )
        if r2.status_code == 200:
            return r2.json()
    except Exception:
        pass

    return verify_payload

def get_member(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("id"), str) and payload.get("id", "").startswith("mem_"):
        return payload
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

def extract_price_ids(payload: Dict[str, Any]) -> List[str]:
    text = json.dumps(payload)
    candidates: List[str] = []
    candidates += re.findall(r"prc_[a-zA-Z0-9\-_]+", text)
    out = []
    seen = set()
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def resolve_tier(price_ids: List[str]) -> str:
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
        return 4
    return 0

def get_selected_lanes_from_payload(payload: Dict[str, Any]) -> List[str]:
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
    out: List[str] = []
    for ln in lanes:
        if ln in LANES_ALL and ln not in out:
            out.append(ln)
    return out

def update_member_custom_fields(member_id: str, custom_fields: Dict[str, Any]) -> bool:
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

def lane_status_from_df(df: Optional[pd.DataFrame], lane_name: str) -> str:
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

# =========================================================
# SSI RATINGS (PUBLIC OUTPUT, PRIVATE LOGIC)
# =========================================================
def ssi_grade(score_1_to_10: float) -> str:
    s = float(score_1_to_10 or 0.0)
    if s >= 9.0: return "SSI-AAA"
    if s >= 8.0: return "SSI-AA"
    if s >= 7.0: return "SSI-A"
    if s >= 6.0: return "SSI-B"
    if s >= 5.0: return "SSI-C"
    if s >= 4.0: return "SSI-D"
    return "SSI-E"

def confidence_descriptor(score_1_to_10: float) -> str:
    s = float(score_1_to_10 or 0.0)
    if s >= 9.0: return "Very High"
    if s >= 8.0: return "High"
    if s >= 7.0: return "Moderate"
    if s >= 6.0: return "Guarded"
    return "Low"

def regime_descriptor(regime_value: Any) -> str:
    r = str(regime_value or "").strip().lower()
    if not r:
        return "Neutral"
    if any(k in r for k in ["strong", "trend"]):
        return "Strong"
    if any(k in r for k in ["weak", "bear", "chop"]):
        return "Weak"
    return "Neutral"

def risk_descriptor(row: Dict[str, Any]) -> str:
    v = row.get("vol", None)
    try:
        if v is not None and v != "":
            vv = float(v)
            if vv >= 7.0:  # NOTE: your vol is 0..10, so keep it consistent
                return "High"
            if vv >= 4.0:
                return "Moderate"
            return "Low"
    except Exception:
        pass
    r = regime_descriptor(row.get("regime"))
    if r == "Weak": return "High"
    if r == "Neutral": return "Moderate"
    return "Low"

def build_scorecard_rows(df: Optional[pd.DataFrame], lane_name: str) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        sym = r.get("symbol", None)
        if sym is None or str(sym).strip() == "":
            continue
        status = str(r.get("status", "ok")).lower().strip()
        try:
            score = float(r.get("ssi", 0) or 0)
        except Exception:
            score = 0.0

        row_dict = r.to_dict() if hasattr(r, "to_dict") else dict(r)
        rows.append(
            {
                "Ticker": str(sym).upper(),
                "SSI-Grade": ssi_grade(score),
                "Confidence": confidence_descriptor(score),
                "Risk": risk_descriptor(row_dict),
                "Regime": regime_descriptor(r.get("regime")),
                "Universe": "SSI Core Universe" if (status == "ok" and score >= 7.0) else "SSI Extended Universe",
                "_lane": lane_name,
                "_score": score,
                "_status": status,
            }
        )
    return rows

# =========================================================
# DAILY CADENCE CACHE
# =========================================================
@st.cache_data(show_spinner=False)
def run_scan_for_date(ny_date: str) -> Dict[str, Any]:
    return engine.run_full_scan()

def get_scan_result(force_refresh: bool) -> Dict[str, Any]:
    key = ny_date_key()
    if force_refresh:
        run_scan_for_date.clear()
    return run_scan_for_date(key)

# =========================================================
# HEADER + RUN
# =========================================================
st.title("SSI Market Decision Engine")
st.caption("Public doctrine. Private engine. Institution-ready output.")

bar_l, bar_r = st.columns([3, 1])
with bar_l:
    st.markdown(
        '<div class="brandbar"><b>SSI Ratings</b><br>'
        '<span class="smallmuted">Free scorecard. Dashboards require an active plan.</span></div>',
        unsafe_allow_html=True
    )
with bar_r:
    force_refresh = st.button("Refresh Now", use_container_width=True)

with st.spinner("Running scan…"):
    res = get_scan_result(force_refresh)

if not res:
    st.error("Scan failed. Try again.")
    st.stop()

headline_ssi = float(res.get("headline_ssi", 0.0) or 0.0)
banner = str(res.get("risk_banner", "Unknown") or "Unknown")

crypto_df = res.get("crypto")
fx_df = res.get("fx")
opt_df = res.get("options")
lc_df = res.get("lc")  # <-- NEW (engine must provide this for LC to populate)

# =========================================================
# PUBLIC SCORECARD (FREE)
# =========================================================
st.markdown("## SSI Daily Legitimacy Scorecard")
st.caption("Free to read. Platform embedding, bulk usage, or redistribution is licensed.")

scorecard_rows: List[Dict[str, Any]] = []
scorecard_rows += build_scorecard_rows(crypto_df, "Crypto")
scorecard_rows += build_scorecard_rows(fx_df, "Forex")
scorecard_rows += build_scorecard_rows(opt_df, "Options")
scorecard_rows += build_scorecard_rows(lc_df, "Long Cycle")

scorecard_df = pd.DataFrame(scorecard_rows)
if scorecard_df.empty:
    st.warning("Scorecard unavailable (no lane data).")
else:
    scorecard_df = scorecard_df.sort_values(by=["_score"], ascending=False)
    scorecard_df = scorecard_df.drop_duplicates(subset=["Ticker"], keep="first")
    scorecard_df["UniverseRank"] = scorecard_df["Universe"].apply(lambda x: 0 if "Core" in x else 1)
    scorecard_df = scorecard_df.sort_values(by=["UniverseRank", "_score"], ascending=[True, False])
    show_cols = ["Ticker", "SSI-Grade", "Confidence", "Risk", "Regime", "Universe"]
    st.dataframe(scorecard_df[show_cols], use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# PUBLIC OVERVIEW (NO LOGIN REQUIRED)
# =========================================================
c1, c2 = st.columns([1, 3])
with c1:
    st.metric("SSI Score", f"{headline_ssi:.2f}")
with c2:
    st.markdown(f"### {banner}")
st.caption(f"Cadence: daily close refresh + event-driven updates. (NY date: {ny_date_key()})")

st.divider()

st.markdown("## Lane Activity")
st.caption("Status only. Log in to unlock dashboards.")
status_df = pd.DataFrame(
    [
        {"Lane": "Crypto", "Status": lane_status_from_df(crypto_df, "Crypto")},
        {"Lane": "Forex", "Status": lane_status_from_df(fx_df, "Forex")},
        {"Lane": "Options", "Status": lane_status_from_df(opt_df, "Options")},
        {"Lane": "Long Cycle", "Status": lane_status_from_df(lc_df, "Long Cycle")},
    ]
)
st.dataframe(status_df, use_container_width=True, hide_index=True)

st.divider()

# =========================================================
# ADMIN BYPASS (OPTIONAL)
# =========================================================
admin_code = get_query_param("admin")
ADMIN_CODE_OK = bool(SSI_ADMIN_CODE) and (admin_code == SSI_ADMIN_CODE)

# =========================================================
# AUTH GATE (DASHBOARDS)
# =========================================================
token = get_query_param("ms")

if ADMIN_CODE_OK:
    tier = "Black"
    max_lanes = 4
    selected_lanes = LANES_ALL[:]
    member_email = "ADMIN"
    member_id = None
    st.success("✅ Admin access enabled (code)")
else:
    if not token:
        st.markdown("## Unlock dashboards")
        st.info("Log in if you already have an account. New? Choose a plan.")
        colA, colB = st.columns(2)
        with colA:
            if LOGIN_URL:
                st.link_button("Log in", LOGIN_URL, use_container_width=True)
            else:
                st.button("Log in (set WEBFLOW_BASE_URL)", disabled=True, use_container_width=True)
        with colB:
            if PLANS_URL:
                st.link_button("Choose a plan", PLANS_URL, use_container_width=True)
            else:
                st.button("Choose a plan (set WEBFLOW_BASE_URL)", disabled=True, use_container_width=True)
        st.stop()

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
    member_email_raw = get_member_email(member_payload) or ""
    member_email = member_email_raw.strip().lower() if member_email_raw else None

    # Admin override (email OR member id)
    if (member_email and member_email in SSI_ADMIN_EMAILS) or (member_id and member_id in SSI_ADMIN_MEMBER_IDS):
        tier = "Black"
        max_lanes = 4

    left, right = st.columns([3, 2])
    with left:
        shown_email = member_email_raw.strip() if member_email_raw else "(email not returned)"
        shown_mid = member_id if member_id else "(member_id not returned)"
        st.success(f"✅ Logged in as {shown_email}")
        st.caption(f"member_id: {shown_mid}")
    with right:
        st.markdown(
            f'<div class="card"><b>Tier:</b> <span class="pill">{tier}</span><br>'
            f'<span class="smallmuted">Lanes allowed:</span> <b>{max_lanes}</b></div>',
            unsafe_allow_html=True,
        )

    if max_lanes == 0:
        st.warning("You are logged in, but you don't have an active plan that unlocks dashboards.")
        if PLANS_URL:
            st.link_button("Choose a plan", PLANS_URL)
        st.stop()

    selected_lanes = get_selected_lanes_from_payload(member_payload)

    if tier == "Black":
        selected_lanes = LANES_ALL[:]
        if member_id:
            update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: selected_lanes})
    else:
        if len(selected_lanes) > max_lanes and member_id:
            trimmed = selected_lanes[:max_lanes]
            ok = update_member_custom_fields(member_id, {CUSTOMFIELD_SELECTED_LANES: trimmed})
            if ok:
                selected_lanes = trimmed

    needs_selection = (tier in ("Starter", "Pro")) and (len(selected_lanes) == 0)
    if tier in ("Starter", "Pro"):
        with st.expander("⚙️ Manage lanes", expanded=needs_selection):
            st.subheader("Configure your engine")
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

    if not selected_lanes:
        st.warning("Choose your lane(s) to unlock dashboards.")
        st.stop()

# =========================================================
# ON-DEMAND SCORING (LOGGED-IN ONLY)
# =========================================================
st.divider()
st.markdown("## Score any symbol (on-demand)")
st.caption("Type any ticker or pair. LC only accepts US equities tickers (no -USD or =X).")

def _is_valid_lc_symbol(sym: str) -> bool:
    s = sym.strip().upper()
    if not s:
        return False
    if "-USD" in s or "=X" in s:
        return False
    # basic sanity: allow letters, numbers, dash, dot
    return bool(re.fullmatch(r"[A-Z0-9\.\-]{1,15}", s))

with st.expander("Score a symbol now", expanded=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        custom_symbol = st.text_input("Symbol", value="", placeholder="Examples: ETH-USD, EURUSD=X, NVDA")
    with col2:
        lane_choice = st.selectbox("Lane", options=selected_lanes)

    if st.button("Score symbol", use_container_width=True):
        sym = (custom_symbol or "").strip().upper()
        if not sym:
            st.warning("Enter a symbol.")
        else:
            if lane_choice == "Long Cycle" and not _is_valid_lc_symbol(sym):
                st.error("Long Cycle only accepts US equities tickers (example: NVDA).")
            else:
                label_map = {
                    "Crypto": "CRYPTO",
                    "Forex": "FOREX",
                    "Options": "OPTIONS",
                    "Long Cycle": "LC",
                }
                label = label_map.get(lane_choice, "LC")

                try:
                    # Uses engine's internal scorer. Works today without needing new APIs.
                    row = engine._score_symbol(label, sym)  # type: ignore[attr-defined]
                    df = pd.DataFrame([row])
                    show_lane(f"On-demand — {lane_choice}", df)
                except Exception as e:
                    st.error(f"Could not score {sym}. Error: {e}")

# =========================================================
# DASHBOARDS — ONLY FOR SELECTED LANES
# =========================================================
st.divider()
st.markdown("## Your dashboards")
st.caption(f"Tier: {tier} | Active lanes: {', '.join(selected_lanes)}")

lane_to_df = {
    "Crypto": crypto_df,
    "Forex": fx_df,
    "Options": opt_df,
    "Long Cycle": lc_df,
}

tabs = st.tabs(selected_lanes)
for i, lane in enumerate(selected_lanes):
    with tabs[i]:
        df = lane_to_df.get(lane)
        if lane == "Options" and is_weekend_ny():
            st.info("Options markets are closed on weekends. Data may reflect the last session.")
        show_lane(f"{lane} Lane", df)

st.divider()
st.markdown("### Reminder")
st.caption("Conditions can shift quickly. Re-check before placing size.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    if PLANS_URL:
        st.link_button("Pricing / Plans", PLANS_URL, use_container_width=True)
with col2:
    if LOGIN_URL:
        st.link_button("Account / Login", LOGIN_URL, use_container_width=True)
with col3:
    st.markdown('<span class="smallmuted">Tip: bookmark this page after login.</span>', unsafe_allow_html=True)