import asyncio
import logging
import traceback

# To avoid any asyncio loop, concurrent blocking operations/issues
import nest_asyncio
import pandas as pd
import streamlit as st
from src.constants import DataFrameKeys
from src.failure_analyzer import FailureAnalyzer
from src.helpers import create_excel_with_clusters, get_tc_ids_from_sql, tc_id_scheduler

nest_asyncio.apply()

################################## Configurations and Global Streamlit Sessions ####################################
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
tc_id_scheduler()
logger = logging.getLogger("Issue Grouping")

st.set_page_config(page_title="Issue Grouping", page_icon=":material/group:", layout="wide")

# Initialize session state variables
if "tc_ids_options" not in st.session_state:
    try:
        with st.spinner("Getting TC IDS for tests from SQL ..."):
            st.session_state.tc_ids_options = get_tc_ids_from_sql()
    except Exception as e:
        traceback.print_exc()
        logger.info(f"Failed to load test case IDs: {str(e)}")
        st.session_state.tc_ids_options = pd.DataFrame()

if "processed_data" not in st.session_state:
    st.session_state.processed_data = False

if "clustered_df" not in st.session_state:
    st.session_state.clustered_df = None

if "last_processed_source" not in st.session_state:
    st.session_state.last_processed_source = {
        "type": None,
        "value": None,
    }  # e.g., {"type": "file", "value": file_name} or {"type": "tc_id", "value": tc_id}

analyzer = FailureAnalyzer()


################################################ Main Page Start ########################################################
st.title("Issue Grouping App 🫂")

# Use a form to collect input and trigger processing
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        uploaded_file_form = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"], key="file_uploader_form")

    with col2:
        tc_options = []
        if not st.session_state.tc_ids_options.empty:
            try:
                tc_options = st.session_state.tc_ids_options["testplan_id"].tolist()
            except Exception:
                traceback.print_exc()
                pass
        selected_tc_id_form = st.selectbox("Select a Test Case ID", options=tc_options, index=None)

    process_button = st.form_submit_button("Process Data")

# Logic to handle form submission and data processing
if process_button:
    # Determine the input source based on form submission
    current_input_type = None
    current_input_value = None
    df_to_process = None

    with st.spinner("loading data"):
        if uploaded_file_form:
            current_input_type = "file"
            current_input_value = uploaded_file_form.name  # Use file name as identifier for checking change
            st.write("✅ File uploaded.")
            df_to_process = analyzer.load_data(st_obj=uploaded_file_form)
        elif selected_tc_id_form and selected_tc_id_form != "":
            current_input_type = "tc_id"
            current_input_value = selected_tc_id_form
            st.write(f"✅ Selected Test Case ID: {selected_tc_id_form}")
            df_to_process = analyzer.load_data(tc_id=selected_tc_id_form)
        else:
            st.warning("Please upload a file or select a Test Case ID to continue.")
            st.session_state.processed_data = False  # Ensure not processed if no valid input
            st.session_state.clustered_df = None
            current_input_type = None  # Reset input type if no valid input
            current_input_value = None

    uploaded_file_form, selected_tc_id_form = None, None  # clear forms inputs after submission

    # Only process if input is valid AND different from last processed input
    if current_input_type and (
        current_input_type != st.session_state.last_processed_source["type"]
        or current_input_value != st.session_state.last_processed_source["value"]
        or not st.session_state.processed_data  # Re-process if not already processed for some reason
    ):
        if df_to_process is None or df_to_process.empty:
            st.error("No failure test cases found in the data source.")
            st.session_state.processed_data = False
            st.session_state.clustered_df = None
            st.session_state.last_processed_source = {"type": None, "value": None}
        else:
            st.subheader("Preview of loaded data: ")
            st.text(f"Shape of Data: {df_to_process.shape}")
            st.dataframe(df_to_process.head(5))

            with st.spinner("Analyzing and grouping data... This may take a moment."):
                try:
                    clustered_df_new = asyncio.run(analyzer.analyze(dataframe=df_to_process))
                    st.session_state.clustered_df = clustered_df_new
                    st.session_state.processed_data = True
                    st.session_state.last_processed_source = {"type": current_input_type, "value": current_input_value}
                except Exception as e:
                    traceback.print_exc()
                    st.error(f"Error during analysis: {str(e)}")
                    st.session_state.processed_data = False
                    st.session_state.clustered_df = None
                    st.session_state.last_processed_source = {"type": None, "value": None}
    elif current_input_type:  # Valid input, but same as last processed, or already processed
        st.info("Data already processed for this input. Displaying results.")

# Display results if processing is complete and clustered_df is available
if (
    st.session_state.processed_data
    and st.session_state.clustered_df is not None
    and DataFrameKeys.cluster_name in st.session_state.clustered_df
):
    st.success("Grouping completed successfully!")
    try:
        clustered_df = st.session_state.clustered_df

        st.subheader("Clustered Data")
        COL_TO_SHOW = [
            "tc_uuid",
            "soc_name",
            "reason",
            "log",
            DataFrameKeys.cluster_name,
        ]

        # Get unique clusters and sort them
        unique_clusters = [str(c) for c in clustered_df[DataFrameKeys.cluster_name].unique().tolist()]
        clusters = sorted(unique_clusters)

        if not clusters:
            st.warning("No clusters were created. The data might be too similar or too different.")
        else:
            st.info(f"Total clusters created: {len(clusters)}")

            # Create tabs for each cluster
            tabs = st.tabs([c for c in clusters])
            for tab, cluster in zip(tabs, clusters):
                with tab:
                    _sub_cluster = clustered_df[clustered_df[DataFrameKeys.cluster_name] == cluster][COL_TO_SHOW]
                    st.subheader(f"{cluster} - Total Rows: {_sub_cluster.shape[0]}")
                    st.dataframe(_sub_cluster)

            # Create Excel with multiple sheets
            try:
                excel_data = create_excel_with_clusters(clustered_df, DataFrameKeys.cluster_name, COL_TO_SHOW)

                # Determine filename for download
                # Use the last processed source info for file naming
                if st.session_state.last_processed_source["type"] == "file":
                    # We store the original file name in last_processed_source["value"]
                    file_name = f"{st.session_state.last_processed_source['value'].replace('.xlsx', '').replace('.xls', '')}_clustered.xlsx"
                elif st.session_state.last_processed_source["type"] == "tc_id":
                    file_name = f"{st.session_state.last_processed_source['value']}_clustered.xlsx"
                else:
                    file_name = "clustered_data.xlsx"  # Fallback

                st.write("**Please wait for a few seconds after clicking the Download button** 🙏")
                st.download_button(
                    label="Download clustered data as Excel",
                    data=excel_data,
                    file_name=file_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except Exception as e:
                traceback.print_exc()
                st.error(f"Error creating Excel file: {str(e)}")

    except Exception as e:
        traceback.print_exc()
        st.error(f"Error displaying results: {str(e)}")
