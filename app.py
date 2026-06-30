import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go

from tabs.forecasting_model import ForecastingTab
from tabs.market_share_impact import MarketImpactTab

# Page configuration
st.set_page_config(
    page_icon= "🐺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Reading file function
def load_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(uploaded_file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")
    
# TOP
st.title("Retail Intelligent Simulation & Forecasting Engine")
st.markdown("Upload monthly sales file - csv or excel format supported")

data_choice = st.radio(
    "What type of data are you combining?",
    ("Total Sales Data", "Invoice Data"),
    horizontal=True
)

if data_choice == "Total Sales Data":
    req_cols = ['Mall', 'Brand Name', 'Brand Code', 'Tenant Code', 'Date', 'Total Gross Amount']
    key, prefix = "macro", "Combined_Sales_Data"
else:
    req_cols = ['Reference ID', 'Date', 'Time', 'Invoice Number', 'Mall Name', 'Tenant Name', 'Till ID', 'Gross Amount', 'Net Amount', 'Total Tax', 'Discount', 'Amount to be Paid', 'Transaction Type', 'Remark']
    key, prefix = "micro", "Master_Invoice_Data"

TEMP_DATA_PATH = "temp_combined_data.pkl"

# *** Data Retaining ***
if os.path.exists(TEMP_DATA_PATH):
    st.success("📦 Found previously uploaded data in local storage!")

    # Load into memory if not already there
    if 'raw_data' not in st.session_state:
        st.session_state['raw_data'] = pd.read_pickle(TEMP_DATA_PATH)
        st.session_state['data_type'] = key
        
    combined_df = st.session_state['raw_data']
    
    st.write(f"-- Total Rows: -- {len(combined_df):,} | -- Columns -- {len(combined_df.columns)}")
    st.dataframe(combined_df.head(20))
    
    # Place buttons side-by-side
    col1, col2 = st.columns([2, 10])
    with col1:
        csv_data = combined_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Combined CSV",
            data=csv_data,
            file_name=f"{prefix}.csv",
            mime="text/csv",
            type="primary"
        )
    with col2:
        if st.button("🗑️ Clear Data & Start Fresh"):
            os.remove(TEMP_DATA_PATH)          # Delete local file
            st.session_state.clear()           # Clear memory
            st.rerun()                         # Refresh the app

    # *** Phase 2: Simulation and forecasting engine ***
    st.divider()
    st.subheader("Simulation & Forecasting")
    
    tab_names = [
                "Forecasting Engine", 
                "Top Brands by Turnover",
                "Top Brands by Number of Transactions",
            ]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        tab_0 = ForecastingTab(combined_df)
        tab_0.render()

    with tabs[1]:
        tab_1 = MarketImpactTab(combined_df)
        tab_1.render()

# *** Upload new data ***
else:
    uploaded_files = st.file_uploader(
        f"Upload multiple Excel or CSV files for {data_choice}", 
        type=["xlsx", "xls", "csv"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        df_list = []
        errors = []
        with st.spinner("Combining files in memory..."):
            for file in uploaded_files:
                try:
                    df = load_file(file)
                    df.columns = df.columns.str.strip()

                    missing_cols = [c for c in req_cols if c not in df.columns]
                    if missing_cols:
                        errors.append(f"⚠️ {file.name} is missing: {', '.join(missing_cols)}")

                    cols_to_keep = [c for c in req_cols if c in df.columns]
                    df = df[cols_to_keep]
                    df_list.append(df)

                except Exception as e:
                    errors.append(f"❌ Error in {file.name}: {e}")

            for err in errors:
                st.warning(err)

            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)

                # Save locally and refresh to trigger Branch 1
                combined_df.to_pickle(TEMP_DATA_PATH)
                st.rerun()