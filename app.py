import asyncio

import streamlit as st

from src.constants import DataFrameKeys
from src.failure_analyzer import FailureAnalyzer
from src.helpers import create_excel_with_clusters

st.set_page_config(page_title="Issue Grouping", page_icon=":material/group:", layout="wide")
analyzer = FailureAnalyzer()
st.title("Issue Grouping App 🫂", )

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

# Initialize session state
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
    st.session_state.processed = False
    st.session_state.clustered_df = None

# Detect new file upload
if uploaded_file is not None and uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.processed = False
    st.session_state.clustered_df = None

# Run pipeline only if file is uploaded and not yet processed
if uploaded_file is not None and not st.session_state.processed:
    try:
        with st.spinner("Loading data !!!"):
            df = analyzer.load_data(uploaded_file)
        if df is None:
            st.error("The uploaded Excel doesn't have any Failure Test Cases !!!")
        else:
            st.dataframe(df.head(5))

            with st.spinner("Analyzing data..."):
                clustered_df = asyncio.run(analyzer.analyze(dataframe=df))
            st.success("Grouping completed successfully!")

            # Save results in session state
            st.session_state.clustered_df = clustered_df
            st.session_state.processed = True

    except Exception as e:
        st.error(e)

# Show results only if processed
if st.session_state.processed and st.session_state.clustered_df is not None:
    try:
        clustered_df = st.session_state.clustered_df

        st.subheader("Clustered Data")
        COL_TO_SHOW = [
            "tc_uuid",
            "soc_name",
            "reason",
            DataFrameKeys.preprocessed_text_key,
            "log",
            DataFrameKeys.cluster_name,
        ]
        clusters = clustered_df[DataFrameKeys.cluster_name].unique()
        st.info(f"Total clusters created: {len(clusters)}")

        tabs = st.tabs([f"{c}" for c in clusters])
        for tab, cluster in zip(tabs, clusters):
            with tab:
                _sub_cluster = clustered_df[clustered_df[DataFrameKeys.cluster_name] == cluster][COL_TO_SHOW]
                st.subheader(f"{cluster} - Total Rows {_sub_cluster.shape[0]}")
                st.dataframe(_sub_cluster)

        # Create Excel with multiple sheets
        excel_data = create_excel_with_clusters(clustered_df, DataFrameKeys.cluster_name, COL_TO_SHOW)

        st.write("**Please wait for few seconds after clicking Download button** 🙏")
        st.download_button(
            label="Download clustered data as Excel",
            data=excel_data,
            file_name=f"{uploaded_file.name}_clustered.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(e)
