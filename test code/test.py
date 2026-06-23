import streamlit as st
import pandas as pd
import os

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

            st.session_state['data_type'] = key
            st.session_state['raw_data'] = combined_df

            st.success(f"Combined {len(uploaded_files)} files in memory!")
            st.write(f"-- Total Rows: -- {len(combined_df): ,} | -- Columns -- {len(combined_df.columns)}")
            st.dataframe(combined_df.head(20))

            csv_data = combined_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Combined CSV",
                data=csv_data,
                file_name=f"{prefix}.csv",
                mime="text/csv",
                type="primary"
            )
            
            # ==========================================
            # TAB BAR (RIBBON) FOR ANALYSIS ENGINES
            # ==========================================
            st.divider()
            st.subheader("⚙️ Simulation & Forecasting Modules")
            
            # Define your tabs here. Add or remove names to expand the ribbon.
            tab_names = [
                "📈 Forecasting Engine", 
                "🏷️ Price Elasticity", 
                "💻 Online Shift", 
                "🌍 Macro Trends", 
                "⚠️ Supply Chain",
                "➕ Add More..."
            ]
            
            # Unpack the tabs dynamically
            tabs = st.tabs(tab_names)
            
            with tabs[0]:
                st.info("Forecasting Engine module will go here. Uses: `combined_df`")
                # Call forecasting function here later
                
            with tabs[1]:
                st.info("Price Elasticity simulator will go here.")
                # Call price elasticity function here later
                
            with tabs[2]:
                st.info("Online Shopping Shift simulator will go here.")
                
            with tabs[3]:
                st.info("Macro Spending Habits simulator will go here.")
                
            with tabs[4]:
                st.info("Supply Chain Disruption simulator will go here.")
                
            with tabs[5]:
                st.info("Placeholder for future modules.")