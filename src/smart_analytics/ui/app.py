import streamlit as st

home_page = st.Page(
    "mainpage.py",
    title="Smart analytics",
    icon="🌐",
    default=True
)

smarteda_page = st.Page(
    "eda_ui.py", 
    title="Smart EDA", 
    icon="📊"
)

pg = st.navigation([home_page, smarteda_page], position="top")

pg.run()