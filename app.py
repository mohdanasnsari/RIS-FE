import streamlit as st
import pandas as pd
import os

# 1. Page Configuration (Desktop Mode)
st.set_page_config(
    page_title="Retail Intelligence Engine", 
    page_icon="🏬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to read either CSV or Excel safely
def load_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == '.csv':
        return pd.read_csv(uploaded_file)
    elif file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")

# 2. Main Header
st.title("🏬 Retail Intelligent Simulation & Forecasting Engine")
st.markdown("Upload your monthly sales files (Supports **CSV**, **XLSX**, and **XLS** formats). The engine will automatically merge them into a single chronological master dataset.")
st.divider()

# 3. Create Side-by-Side Layout for the Two Data Pipelines
col1, col2 = st.columns(2)

# --- OPTION 1: MACRO AGGREGATED DATA PIPELINE ---
with col1:
    st.subheader("6️⃣ Option 1: Macro Aggregated Data")
    st.caption("Upload multiple monthly files containing macro sales data.")
    
    st.info("""
    **Required Columns per File:**
    `Mall`, `Brand Name`, `Brand Code`, `Tenant Code`, `Date`, `Total Gross Amount`
    """)
    
    # Updated type parameter to include Excel formats
    macro_files = st.file_uploader(
        "Upload Macro Files (Select multiple CSV/Excel sheets)", 
        type=['csv', 'xlsx', 'xls'], 
        accept_multiple_files=True,
        key="macro_uploader"
    )
    
    if macro_files:
        all_macro_dfs = []
        errors = []
        
        for file in macro_files:
            try:
                df = load_file(file)
                # Quick column validation check
                required_cols = {'Mall', 'Brand Name', 'Brand Code', 'Tenant Code', 'Date', 'Total Gross Amount'}
                if not required_cols.issubset(df.columns):
                    errors.append(f"⚠️ {file.name} is missing required macro columns.")
                else:
                    all_macro_dfs.append(df)
            except Exception as e:
                errors.append(f"❌ Error reading {file.name}: {e}")
        
        for err in errors:
            st.error(err)
            
        if all_macro_dfs:
            master_macro_df = pd.concat(all_macro_dfs, ignore_index=True)
            
            # Save to Streamlit session memory
            st.session_state['data_type'] = 'macro'
            st.session_state['raw_data'] = master_macro_df
            
            st.success(f"✅ Successfully combined {len(all_macro_dfs)} monthly file(s) into memory!")
            st.metric(label="Total Combined Rows", value=f"{len(master_macro_df):,}")
            
            st.markdown("**Combined Dataset Preview (First 5 Rows):**")
            st.dataframe(master_macro_df.head(5), use_container_width=True)


# --- OPTION 2: MICRO INVOICE DATA PIPELINE ---
with col2:
    st.subheader("🧾 Option 2: Micro Invoice Data")
    st.caption("Upload multiple monthly files containing raw transactional invoices.")
    
    st.warning("""
    **Required Columns per File:**
    `Reference ID`, `Date`, `Time`, `Invoice Number`, `Mall Name`, `Tenant Name`, `Till ID`, `Gross Amount`, `Net Amount`, `Total Tax`, `Discount`, `Amount to be Paid`, `Transaction Type`, `Remark`
    """)
    
    # Updated type parameter to include Excel formats
    micro_files = st.file_uploader(
        "Upload Invoice Files (Select multiple CSV/Excel sheets)", 
        type=['csv', 'xlsx', 'xls'], 
        accept_multiple_files=True,
        key="micro_uploader"
    )
    
    if micro_files:
        all_micro_dfs = []
        errors = []
        
        for file in micro_files:
            try:
                df = load_file(file)
                # Quick column validation check for micro invoice format
                if 'Invoice Number' not in df.columns or 'Tenant Name' not in df.columns:
                    errors.append(f"⚠️ {file.name} does not match the invoice schema.")
                else:
                    all_micro_dfs.append(df)
            except Exception as e:
                errors.append(f"❌ Error reading {file.name}: {e}")
                
        for err in errors:
            st.error(err)
            
        if all_micro_dfs:
            master_micro_df = pd.concat(all_micro_dfs, ignore_index=True)
            
            # Save to Streamlit session memory
            st.session_state['data_type'] = 'micro'
            st.session_state['raw_data'] = master_micro_df
            
            st.success(f"✅ Successfully combined {len(all_micro_dfs)} monthly invoice file(s) into memory!")
            st.metric(label="Total Combined Transactions", value=f"{len(master_micro_df):,}")
            
            st.markdown("**Combined Invoice Preview (First 5 Rows):**")
            st.dataframe(master_micro_df.head(5), use_container_width=True)