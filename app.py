# app.py
import streamlit as st
import engine

st.set_page_config(page_title="SSI Market Decision Engine", layout="wide")

st.title("SSI Market Decision Engine")
st.caption("Crypto + FX + Options scan → simple SSI regime + lane recommendations")

run = st.button("Run Full Scan")

if run:
    crypto_df = engine.run_crypto_scan()
    fx_df = engine.run_fx_scan()
    opt_df = engine.run_options_scan()

    ssi = engine.compute_ssi(crypto_df, fx_df, opt_df)
    headline = engine.recommend_lane(ssi, crypto_df, fx_df, opt_df)

    st.subheader("Market Regime")
    st.metric("SSI Score", round(float(ssi), 2))
    st.success(headline)

    # Crypto
    st.subheader("Crypto Lane")
    st.dataframe(crypto_df, use_container_width=True)
    cmsg = engine.recommend_crypto(crypto_df)
    if cmsg:
        st.info(cmsg)

    # Forex
    st.subheader("Forex Lane")
    st.dataframe(fx_df, use_container_width=True)
    fmsg = engine.recommend_fx(fx_df)
    if fmsg:
        st.info(fmsg)

    # Options
    st.subheader("Options Lane")
    st.dataframe(opt_df, use_container_width=True)

    orec = engine.recommend_options_contract(opt_df)
    if orec:
        st.warning(
            f"{orec['strategy']} | {orec['symbol']} | Bias: {orec['bias']} | "
            f"Exp: {orec['expiry']} | Strike: {orec['strike']} | Est premium: {orec.get('est_premium','N/A')}"
        )
    else:
        st.caption("No options contract recommendation triggered (score below internal threshold).")

else:
    st.info("Tap **Run Full Scan** to generate the crypto / forex / options outputs.")