import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime

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
            ]
    tabs = st.tabs(tab_names)
    
    # !***! !***! !***! !***! !***! !***! !***! !***! !***!
    with tabs[0]:
        st.info("Forecasting Model - Data Prep & Aggregation")
        
        # --- 1. SILENT DATA CLEANING & FEATURE ENGINEERING ---
        with st.spinner("Cleaning data and extracting time features..."):
            initial_rows = len(combined_df)
            
            # Cleaning
            combined_df = combined_df.drop_duplicates()
            combined_df['Total Gross Amount'] = pd.to_numeric(combined_df['Total Gross Amount'], errors='coerce')
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
            combined_df = combined_df.dropna(subset=['Date', 'Total Gross Amount'])
            
            # Feature Extraction
            combined_df['Day_Name'] = combined_df['Date'].dt.day_name()
            combined_df['Day_Number'] = combined_df['Date'].dt.dayofweek
            combined_df['Is_Weekend'] = combined_df['Day_Number'].isin([5, 6]).astype(int)
            combined_df['Month'] = combined_df['Date'].dt.month
            combined_df['Year'] = combined_df['Date'].dt.year

        # --- 2. ESSENTIAL METRICS ONLY ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Valid Transactions", f"{len(combined_df):,}", f"{initial_rows - len(combined_df)} invalid dropped", delta_color="off")
        col2.metric("Unique Dates", combined_df['Date'].nunique())
        col3.metric("Date Range", f"{combined_df['Date'].dt.date.min()} to {combined_df['Date'].dt.date.max()}")

        st.divider()

        # --- 3. DYNAMIC FILTERING & AGGREGATION ---
        st.subheader("Scope Selection")
        
        mall_col = 'Mall' if 'Mall' in combined_df.columns else ('Mall Name' if 'Mall Name' in combined_df.columns else None)
        brand_col = 'Brand Name' if 'Brand Name' in combined_df.columns else ('Tenant Name' if 'Tenant Name' in combined_df.columns else None)
        filtered_df = combined_df.copy()

        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            if mall_col:
                malls = ["All"] + sorted(list(combined_df[mall_col].dropna().unique()))
                selected_malls = st.multiselect("🏢 Filter by Mall(s)", malls, default=["All"])
                if "All" not in selected_malls and selected_malls:
                    filtered_df = filtered_df[filtered_df[mall_col].isin(selected_malls)]

        with col_f2:
            if brand_col:
                available_brands = ["All"] + sorted(list(filtered_df[brand_col].dropna().unique()))
                selected_brands = st.multiselect("🏷️ Filter by Brand(s)", available_brands, default=["All"])
                if "All" not in selected_brands and selected_brands:
                    filtered_df = filtered_df[filtered_df[brand_col].isin(selected_brands)]

        # Aggregate the FILTERED data
        with st.spinner("Aggregating daily sales..."):
            daily_sales = filtered_df.groupby('Date')['Total Gross Amount'].sum().reset_index()
            daily_sales['Day_Name'] = daily_sales['Date'].dt.day_name()
            daily_sales['Day_Number'] = daily_sales['Date'].dt.dayofweek
            daily_sales['Is_Weekend'] = daily_sales['Day_Number'].isin([5, 6]).astype(int)
            daily_sales['Month'] = daily_sales['Date'].dt.month
            daily_sales['Year'] = daily_sales['Date'].dt.year
            
            # Context columns
            daily_sales['Scope_Mall'] = "All" if "All" in selected_malls else ", ".join(selected_malls)
            daily_sales['Scope_Brand'] = "All" if "All" in selected_brands else ", ".join(selected_brands)
            
            cols = ['Date', 'Scope_Mall', 'Scope_Brand', 'Total Gross Amount', 'Day_Name', 'Is_Weekend', 'Month', 'Year', 'Day_Number']
            daily_sales = daily_sales[cols]
            
        # --- 4. FINAL CLEAN OUTPUT ---
        st.success("✅ Data ready for model training.")
        
        col_agg1, col_agg2 = st.columns([1, 3])
        with col_agg1:
            st.metric("Total Scope Revenue", f"₹ {daily_sales['Total Gross Amount'].sum():,.2f}")
            st.metric("Total Days Modeled", len(daily_sales))
        with col_agg2:
            st.dataframe(daily_sales.head(8), use_container_width=True)
            
        st.session_state['daily_sales'] = daily_sales

        # ==========================================
        # AI FORECASTING ENGINE
        # ==========================================
        st.divider()
        st.subheader("🤖 AI Forecasting Engine")
        
        if 'daily_sales' in st.session_state:
            df = st.session_state['daily_sales']
            
            # 1. Train the Model
            with st.spinner("Training Random Forest Model..."):
                # Add Day_of_Month as required by your model
                if 'Day_of_Month' not in df.columns:
                    df['Day_of_Month'] = df['Date'].dt.day
                    
                features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
                
                X_train = df[features]
                y_train = df['Total Gross Amount']
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
            st.success("✅ Model trained successfully on the filtered scope!")
            
            # 2. Prediction Inputs
            st.markdown("### Generate Future Forecast")
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                target_month = st.number_input("Target Month (1-12)", min_value=1, max_value=12, value=datetime.date.today().month)
            with col_p2:
                target_year = st.number_input("Target Year", min_value=2020, max_value=2100, value=datetime.date.today().year + 1)
            with col_p3:
                pred_type = st.selectbox("Forecast Type", ["Full Month", "Weekends Only", "Weekdays Only", "Custom Dates"])
                
            custom_dates_str = ""
            if pred_type == "Custom Dates":
                custom_dates_str = st.text_input("Enter dates separated by comma (e.g., 1, 15, 25):", "1, 15, 25")

            # 3. Execution Engine
            if st.button("🚀 Run Forecast", type="primary"):
                with st.spinner("Generating predictions..."):
                    # Generate Future Features
                    start_date = f"{target_year}-{target_month:02d}-01"
                    end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
                    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    
                    pred_df = pd.DataFrame({'Date': future_dates})
                    pred_df['Day_Name'] = pred_df['Date'].dt.day_name()
                    pred_df['Day_Number'] = pred_df['Date'].dt.dayofweek
                    pred_df['Day_of_Month'] = pred_df['Date'].dt.day
                    pred_df['Is_Weekend'] = pred_df['Day_Number'].isin([5, 6]).astype(int)
                    pred_df['Month'] = pred_df['Date'].dt.month
                    pred_df['Year'] = pred_df['Date'].dt.year
                    
                    # Apply specific filters based on user selection
                    if pred_type == "Weekends Only":
                        pred_df = pred_df[pred_df['Is_Weekend'] == 1]
                    elif pred_type == "Weekdays Only":
                        pred_df = pred_df[pred_df['Is_Weekend'] == 0]
                    elif pred_type == "Custom Dates" and custom_dates_str:
                        try:
                            requested_days = [int(day.strip()) for day in custom_dates_str.split(',')]
                            pred_df = pred_df[pred_df['Day_of_Month'].isin(requested_days)]
                        except ValueError:
                            st.error("Invalid custom dates format. Please use numbers separated by commas.")
                            st.stop()
                    
                    if pred_df.empty:
                        st.warning("No dates match your criteria.")
                        st.stop()
                    
                    # Predict
                    pred_df['Predicted_Sales'] = model.predict(pred_df[features]).round(2)
                    total_predicted = pred_df['Predicted_Sales'].sum()
                    
                    # Display Results
                    st.success(f"Forecast Complete for {start_date[:7]}")
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.metric(f"Total Predicted ({pred_type})", f"₹ {total_predicted:,.2f}")
                        st.dataframe(pred_df[['Date', 'Day_Name', 'Predicted_Sales']].reset_index(drop=True), use_container_width=True)
                        
                    with c2:
                        # 4. Graphing Logic (Adapted from your script)
                        pred_df['Day'] = pred_df['Date'].dt.day
                        
                        fig, ax = plt.subplots(figsize=(10, 5))
                        box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')
                        
                        plot_color = 'blue' if pred_type in ["Full Month", "Weekdays Only"] else 'orange'
                        marker_type = 'o' if pred_type in ["Full Month", "Weekdays Only"] else 's'
                        
                        # Use Bar for custom/weekends, Line for continuous days
                        if pred_type in ["Weekends Only", "Custom Dates"]:
                            bars = ax.bar(pred_df['Day'], pred_df['Predicted_Sales'], color=plot_color, width=0.6)
                            for bar in bars:
                                yval = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2, yval + (pred_df['Predicted_Sales'].max() * 0.02), f"{yval:,.0f}", ha='center', va='bottom', fontsize=8, color=plot_color, rotation=90)
                        else:
                            ax.plot(pred_df['Day'], pred_df['Predicted_Sales'], color=plot_color, marker=marker_type, linewidth=2)
                            for x, y in zip(pred_df['Day'], pred_df['Predicted_Sales']):
                                ax.text(x, y + (pred_df['Predicted_Sales'].max() * 0.02), f"{y:,.0f}", ha='center', va='bottom', fontsize=8, color=plot_color, rotation=90)
                                
                        ax.set_title(f"Predicted Future Sales: {start_date[:7]} ({pred_type})", fontsize=13, fontweight='bold', pad=10)
                        ax.set_ylabel('Sales Amount (₹)', fontsize=12)
                        ax.set_xlabel('Day of the Month', fontsize=12)
                        ax.set_xticks(pred_df['Day'])
                        
                        ymin, ymax = ax.get_ylim()
                        ax.set_ylim(0 if pred_type != "Full Month" else ymin, ymax * 1.35) 
                        ax.grid(True, alpha=0.3)
                        
                        # Render the matplotlib figure in Streamlit
                        st.pyplot(fig)


# !===! !===! !===! !===! !===! !===! !===! !===! !===! !===! !===! !===! !===! !===! !===! !===!

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