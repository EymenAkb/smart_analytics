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

timeseries_page = st.Page(
    "time_ui.py",
    title="Time Series",
    icon="📈"
)

pg = st.navigation([home_page, smarteda_page, timeseries_page], position="top")

pg.run()