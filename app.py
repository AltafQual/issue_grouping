import streamlit as st

pages = {
    "Issue Grouping": [
        st.Page("streamlit_pages/issue_grouping.py", title="Issue Grouping"),
    ],
    "Regression": [st.Page("streamlit_pages/regression.py", title="Regression")],
    "Error Classification": [st.Page("streamlit_pages/error_classification.py", title="Error Classification")],
}

pg = st.navigation(pages)
pg.run()
