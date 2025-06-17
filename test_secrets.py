import streamlit as st

st.title("🔐 Test Streamlit Secret")
st.write("Your OpenAI API Key is:")
st.code(st.secrets["OPENAI_API_KEY"])
