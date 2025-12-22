# app.py
import streamlit as st
import engine

st.set_page_config(page_title="SSI Market Decision Engine", layout="centered")

st.title("SSI Market\nDecision Engine")
st.caption("Crypto + FX + Options scan → simple SSI regime + lane recommendations")

if st.button("Run Full Scan"):
    st.session_state["run"] = True

if "run" not in st.session_state:
    st.session_state["run"] = True  # auto-run on load

if st.session_state["run"]:
    result = engine.run_full_scan()

    st.subheader("SSI Score")
    st.metric(label="SSI Score", value=result["headline_ssi"])

    st.info(result["risk_banner"])

    st.subheader("Crypto Lane")
    crypto_df = result["crypto"]
    st.dataframe(crypto_df[["symbol", "last", "trend", "vol", "ssi", "regime", "reco", "status"]], use_container_width=True)

    st.subheader("Forex Lane")
    fx_df = result["fx"]
    st.dataframe(fx_df[["symbol", "last", "trend", "vol", "ssi", "regime", "reco", "status"]], use_container_width=True)

    st.subheader("Options Lane")
    opt_df = result["options"]
    st.dataframe(opt_df[["symbol", "last", "trend", "vol", "ssi", "regime", "reco", "status"]], use_container_width=True)

    # show top recommendation only (clean)
    st.subheader("Top Play Right Now")
    best_rows = []
    for df in [crypto_df, fx_df, opt_df]:
        if not df.empty:
            best_rows.append(df.iloc[0].to_dict())
    best_rows = [r for r in best_rows if r.get("status") == "ok"]

    if best_rows:
        best = sorted(best_rows, key=lambda r: r.get("ssi", 0), reverse=True)[0]
        if best.get("reco"):
            st.success(best["reco"])
        else:
            st.warning("No lane cleared the internal threshold for a recommendation.")
    else:
        st.error("No usable data returned from provider. Check 'status' column per lane.")

with st.expander("Debug (for when scores are zero)"):
    # This will immediately tell us if yfinance returned data or not.
    r = engine.run_full_scan()
    st.write("Headline SSI:", r["headline_ssi"])
    st.write("Crypto status counts:", r["crypto"]["status"].value_counts(dropna=False).to_dict())
    st.write("FX status counts:", r["fx"]["status"].value_counts(dropna=False).to_dict())
    st.write("Options status counts:", r["options"]["status"].value_counts(dropna=False).to_dict())