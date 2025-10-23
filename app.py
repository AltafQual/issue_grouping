import streamlit as st

pg = st.navigation([
    st.Page("streamlit_pages/issue_grouping.py", title="Issue Grouping", icon="🫂"),
    st.Page("streamlit_pages/regression.py", title="Regression", icon="⚔️"),
    st.Page("streamlit_pages/error_classification.py", title="Error Classification", icon="🧐"),
    
])
pg.run()
