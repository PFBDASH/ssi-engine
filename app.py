# app.py
import streamlit as st
import engine

st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

st.title("SSI Market Decision Engine")
st.caption("Crypto + FX + Options decision engine")

if st.button("Run Full Scan"):
    crypto_df = engine.run_crypto_scan()
    fx_df = engine.run_fx_scan()
    options_df = engine.run_options_scan()

    ssi_score = engine.compute_ssi(crypto_df, fx_df, options_df)
    headline = engine.recommend_lane(ssi_score, crypto_df, fx_df, options_df)

    st.subheader("Market Regime")
    st.metric("SSI Score", round(float(ssi_score), 2))
    st.success(headline)

    st.subheader("Crypto Lane")
    st.dataframe(crypto_df, use_container_width=True)
    rec = engine.recommend_crypto(crypto_df)
    if rec:
        st.info(rec)

    st.subheader("Forex Lane")
    st.dataframe(fx_df, use_container_width=True)
    rec = engine.recommend_fx(fx_df)
    if rec:
        st.info(rec)

    st.subheader("Options Lane")
    st.dataframe(options_df, use_container_width=True)
    orec = engine.recommend_options_contract(options_df)
    if orec:
        st.warning(
            f"{orec['strategy']} | {orec['symbol']} | Bias: {orec['bias']} | "
            f"Exp: {orec['expiry']} | Strike: {orec['strike']} | Est premium: {orec.get('est_premium','N/A')}"
        )
else:
    st.info("Tap **Run Full Scan**.")