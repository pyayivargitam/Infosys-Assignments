import streamlit as st

st.title("Hello from Streamlit 🎉")
st.write("Hi Pranav — your Streamlit setup is working perfectly!")
x = st.slider("Pick a number", 0, 100, 50)
st.write("You picked:", x)
