import streamlit as st
import os
from src.failure_analyzer import FailureAnalyzer

analyzer = FailureAnalyzer()
st.title("Issue Grouping App 🫂")

# File upload
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
if uploaded_file is not None:
    try:
        with st.spinner("Loading data !!!"):
            df = analyzer.load_data(uploaded_file)
        if df is None:
            st.error("The uploaded CSV doesn't have any Failure Test Cases !!!")
        else:
            st.dataframe(df.head(5))
            clustered_df = analyzer.analyze(dataframe=df)
            st.success("Grouping completed successfully!")

            # Display the clustered data
            st.subheader("Clustered Data")

            cluster_column = "cluster"  # Change this to your actual column name
            COL_TO_SHOW = ["tc_uuid", "soc_name", "reason", "log"]
            clusters = clustered_df[cluster_column].unique()
            st.info(f"Total clusters created: {len(clusters)}")
            
            # Create a tab for each cluster
            tabs = st.tabs([f"Cluster {c}" for c in clusters])
            for tab, cluster in zip(tabs, clusters):
                with tab:
                    _sub_cluster = clustered_df[clustered_df[cluster_column] == cluster][COL_TO_SHOW]
                    st.subheader(f"Cluster {cluster} - Total Rows {_sub_cluster.shape[0]}")
                    st.dataframe(_sub_cluster)

            # # Download the clustered data as CSV
            csv = clustered_df.to_csv(index=False)
            st.download_button(
                label="Download clustered data as CSV",
                data=csv,
                file_name=f"{os.path.splitext(uploaded_file.name)[0]}_clustered.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(e)
