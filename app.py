import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go

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
        st.info("Forecasting Model - Data Prep & Aggregation")

        data_type = st.session_state.get('data_type', 'macro')
        if 'Total Gross Amount' in combined_df.columns:
            mall_col, brand_col, sales_col = 'Mall', 'Brand Name', 'Total Gross Amount'
        elif 'Gross Amount' in combined_df.columns:
            mall_col, brand_col, sales_col = 'Mall Name', 'Tenant Name', 'Gross Amount'
        else:
            st.error("🚨 Unrecognized data format. Could not find 'Total Gross Amount' or 'Gross Amount'. Please clear data and re-upload.")
            st.stop()

        # --- 1. SILENT DATA CLEANING & FEATURE ENGINEERING ---
        with st.spinner("Cleaning data and extracting time features..."):
            initial_rows = len(combined_df)

            # Cleaning
            combined_df = combined_df.drop_duplicates()
            # Handle potential commas in string amounts before converting to numeric
            combined_df[sales_col] = pd.to_numeric(combined_df[sales_col].astype(str).str.replace(',', ''), errors='coerce')
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='mixed', dayfirst=True, errors='coerce')
            
            # Drop invalid rows based on the dynamic sales column
            combined_df = combined_df.dropna(subset=['Date', sales_col])
            
            # Feature Extraction
            combined_df['Day_Name'] = combined_df['Date'].dt.day_name()
            combined_df['Day_Number'] = combined_df['Date'].dt.dayofweek
            combined_df['Is_Weekend'] = combined_df['Day_Number'].isin([5, 6]).astype(int)
            combined_df['Month'] = combined_df['Date'].dt.month
            combined_df['Year'] = combined_df['Date'].dt.year
            combined_df['Day_of_Month'] = combined_df['Date'].dt.day

            # --- 2. ESSENTIAL METRICS ONLY ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Valid Transactions", f"{len(combined_df):,}", f"{initial_rows - len(combined_df)} invalid dropped", delta_color="off")
            col2.metric("Unique Dates", combined_df['Date'].nunique())
            col3.metric("Date Range", f"{combined_df['Date'].dt.date.min()} to {combined_df['Date'].dt.date.max()}")

            st.divider()

        # --- 3. DYNAMIC CASCADING FILTERING ---
        st.subheader("Scope Selection")
        
        col_f1, col_f2 = st.columns(2)
        
        # Mall Filter
        with col_f1:
            if mall_col in combined_df.columns:
                malls = ["All"] + sorted(list(combined_df[mall_col].dropna().unique()))
                selected_malls = st.multiselect("🏢 Filter by Mall(s)", malls, default=["All"])
            else:
                selected_malls = ["All"]
                st.warning("No Mall column found in this dataset.")

        # Cascading Logic: Filter dataset by selected malls *before* populating Brands
        temp_filtered_df = combined_df.copy()
        if "All" not in selected_malls and selected_malls:
            temp_filtered_df = temp_filtered_df[temp_filtered_df[mall_col].isin(selected_malls)]

        # Brand Filter (Populated dynamically based on Mall selection)
        with col_f2:
            if brand_col in temp_filtered_df.columns:
                available_brands = ["All"] + sorted(list(temp_filtered_df[brand_col].dropna().unique()))
                selected_brands = st.multiselect("🏷️ Filter by Brand(s)", available_brands, default=["All"])
            else:
                selected_brands = ["All"]
                st.warning("No Brand column found in this dataset.")

        # Final filtering based on Brand
        filtered_df = temp_filtered_df.copy()
        if "All" not in selected_brands and selected_brands:
            filtered_df = filtered_df[filtered_df[brand_col].isin(selected_brands)]


        # --- 4. AGGREGATION ---
        with st.spinner("Aggregating daily sales..."):
            # Grouping dynamically by the appropriate sales column
            daily_sales = filtered_df.groupby('Date')[sales_col].sum().reset_index()
            daily_sales.rename(columns={sales_col: 'Total Gross Amount'}, inplace=True) # Normalize for model

        # Re-apply date features to the grouped dataset
            daily_sales['Day_Name'] = daily_sales['Date'].dt.day_name()
            daily_sales['Day_Number'] = daily_sales['Date'].dt.dayofweek
            daily_sales['Is_Weekend'] = daily_sales['Day_Number'].isin([5, 6]).astype(int)
            daily_sales['Month'] = daily_sales['Date'].dt.month
            daily_sales['Year'] = daily_sales['Date'].dt.year
            daily_sales['Day_of_Month'] = daily_sales['Date'].dt.day
            
            # Context columns
            daily_sales['Scope_Mall'] = "All" if "All" in selected_malls else ", ".join(selected_malls)
            daily_sales['Scope_Brand'] = "All" if "All" in selected_brands else ", ".join(selected_brands)
            
            cols = ['Date', 'Scope_Mall', 'Scope_Brand', 'Total Gross Amount', 'Day_Name', 'Is_Weekend', 'Month', 'Year', 'Day_Number', 'Day_of_Month']
            daily_sales = daily_sales[cols]

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
        
        if 'daily_sales' in st.session_state and not st.session_state['daily_sales'].empty:
            df = st.session_state['daily_sales']
            
            # 1. Train the Model
            with st.spinner("Training Random Forest Model..."):
                features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
                
                X_train = df[features]
                y_train = df['Total Gross Amount']
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
            st.success("✅ Model trained successfully on the filtered scope!")

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
                        # 4. Graphing Logic (Upgraded to Plotly for a cheerful & professional look)
                        fig = go.Figure()
                        
                        is_continuous = pred_type in ["Full Month", "Weekdays Only"]
                        
                        # Cheerful modern color palette (Vibrant Blue for continuous, Warm Orange for discrete)
                        main_color = '#3b82f6' if is_continuous else '#f97316' 
                        
                        if is_continuous:
                            # Smooth line chart with a soft gradient fill below
                            fig.add_trace(go.Scatter(
                                x=pred_df['Date'], 
                                y=pred_df['Predicted_Sales'],
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color=main_color, width=3, shape='spline'), # 'spline' makes the line smooth/cheerful
                                marker=dict(size=8, color='white', line=dict(width=2, color=main_color)),
                                fill='tozeroy',
                                fillcolor='rgba(59, 130, 246, 0.1)', # Soft transparent fill
                                hovertemplate='<b>%{x|%d %b %Y}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>'
                            ))
                        else:
                            # Clean bar chart with auto-formatted data labels
                            fig.add_trace(go.Bar(
                                x=pred_df['Date'], 
                                y=pred_df['Predicted_Sales'],
                                name='Forecast',
                                marker_color=main_color,
                                marker_line_color=main_color,
                                marker_line_width=1.5,
                                opacity=0.9,
                                text=pred_df['Predicted_Sales'].apply(lambda x: f'₹{x:,.0f}'),
                                textposition='outside',
                                hovertemplate='<b>%{x|%d %b %Y}</b><br>Predicted: ₹%{y:,.2f}<extra></extra>'
                            ))
                            
                        # Professional layout and styling
                        fig.update_layout(
                            title=dict(
                                text=f"<b>Predicted Future Sales: {start_date[:7]}</b> <span style='font-size: 14px; color: gray;'>({pred_type})</span>",
                                font=dict(size=20, color='#1f2937'),
                                y=0.95
                            ),
                            xaxis_title="",
                            yaxis_title="Sales Amount (₹)",
                            plot_bgcolor='rgba(0,0,0,0)', # Transparent professional background
                            paper_bgcolor='rgba(0,0,0,0)',
                            hovermode="x unified",
                            margin=dict(l=0, r=0, t=50, b=0),
                            xaxis=dict(
                                showgrid=False, 
                                tickformat="%d %b",
                                showline=True, 
                                linewidth=1, 
                                linecolor='#e5e7eb'
                            ),
                            yaxis=dict(
                                showgrid=True, 
                                gridcolor='#f3f4f6', 
                                zeroline=False,
                                tickprefix="₹"
                            ),
                            showlegend=False
                        )
                        
                        # Expand Y-axis slightly for Bar charts so text doesn't get cut off
                        if not is_continuous:
                            fig.update_yaxes(range=[0, pred_df['Predicted_Sales'].max() * 1.2])
                        
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}) # Hides the distracting top toolbar

            else:
                st.warning("Not enough data to train the model. Please adjust your filters.")   

# !***! !***! !***! !***! !***! !***! !***! !***! !***!

    with tabs[1]:
        st.header("Market Share Impact")
        st.markdown("Use actual historical data to see how the arrival of a new brand impacted existing competitors.")

        # Ensure we have the dynamic column names from Tab 0
        if 'Total Gross Amount' in combined_df.columns:
            mall_col, brand_col, sales_col = 'Mall', 'Brand Name', 'Total Gross Amount'
        else:
            mall_col, brand_col, sales_col = 'Mall Name', 'Tenant Name', 'Gross Amount'

        # --- 1. MALL SELECTION ---
        malls = sorted(list(combined_df[mall_col].dropna().unique()))
        selected_mall = st.selectbox("🏢 Select Mall for Analysis", malls, key="impact_mall_select")
        
        mall_df = combined_df[combined_df[mall_col] == selected_mall].copy()

        # --- 2. BRAND ENTRY TIMELINE ---
        # Calculate the very first date each brand recorded a transaction
        arrival_dates = mall_df.groupby(brand_col)['Date'].min().reset_index()
        arrival_dates.columns = ['Brand Name', 'First Arrival Date']
        arrival_dates = arrival_dates.sort_values('First Arrival Date').reset_index(drop=True)

        st.divider()
        st.subheader(f"Brand Entry Timeline: {selected_mall}")
        
        col_list, col_selectors = st.columns([1.2, 2])
        
        with col_list:
            st.caption("Review arrival dates to match competitors.")
            # Format date beautifully for the table
            display_dates = arrival_dates.copy()
            display_dates['First Arrival Date'] = display_dates['First Arrival Date'].dt.strftime('%d %b %Y')
            st.dataframe(display_dates, use_container_width=True, hide_index=True, height=250)

        # --- 3. THE PRE/POST SELECTION ---
        with col_selectors:
            st.caption("Select your comparison targets.")
            incumbent_brand = st.selectbox("🛡️ Select Incumbent Brand (The older brand)", arrival_dates['Brand Name'].tolist())
            entrant_brand = st.selectbox("⚔️ Select New Entrant (The competitor)", arrival_dates['Brand Name'].tolist())
            
            # Get the exact entry date of the competitor
            entry_date = arrival_dates.loc[arrival_dates['Brand Name'] == entrant_brand, 'First Arrival Date'].values[0]
            entry_date = pd.to_datetime(entry_date)
            
            incumbent_start = arrival_dates.loc[arrival_dates['Brand Name'] == incumbent_brand, 'First Arrival Date'].values[0]
            
            if pd.to_datetime(incumbent_start) >= entry_date and incumbent_brand != entrant_brand:
                st.warning("⚠️ The Incumbent brand arrived ON or AFTER the Entrant brand. Please select an older incumbent to see a valid 'Before & After' impact.")

        # --- 4. THE ANALYSIS ENGINE ---
        if incumbent_brand and entrant_brand and incumbent_brand != entrant_brand and pd.to_datetime(incumbent_start) < entry_date:
            st.divider()
            
            # Check if we have Invoice Data for advanced behavioral analysis
            has_invoices = 'Invoice Number' in mall_df.columns
            
            if has_invoices:
                # Group by Date: Sum the Sales, Count unique Invoices
                incumbent_df = mall_df[mall_df[brand_col] == incumbent_brand].groupby('Date').agg(
                    Daily_Sales=(sales_col, 'sum'),
                    Daily_Volume=('Invoice Number', 'nunique')
                ).reset_index()
                # Calculate Daily Average Transaction Value (ATV)
                incumbent_df['Daily_ATV'] = incumbent_df['Daily_Sales'] / incumbent_df['Daily_Volume']
            else:
                # Basic grouping for Macro data
                incumbent_df = mall_df[mall_df[brand_col] == incumbent_brand].groupby('Date').agg(
                    Daily_Sales=(sales_col, 'sum')
                ).reset_index()

            # Sort by date and calculate a 7-day rolling average for the main sales trend
            incumbent_df = incumbent_df.sort_values('Date')
            incumbent_df['7D_Rolling_Avg'] = incumbent_df['Daily_Sales'].rolling(window=7, min_periods=1).mean()
            
            # Split data into Before and After (45-day window)
            window_days = 45
            before_df = incumbent_df[(incumbent_df['Date'] >= (entry_date - pd.Timedelta(days=window_days))) & (incumbent_df['Date'] < entry_date)]
            after_df = incumbent_df[(incumbent_df['Date'] >= entry_date) & (incumbent_df['Date'] <= (entry_date + pd.Timedelta(days=window_days)))]
            
            # --- METRICS CALCULATIONS ---
            # 1. Revenue
            rev_before = before_df['Daily_Sales'].mean() if not before_df.empty else 0
            rev_after = after_df['Daily_Sales'].mean() if not after_df.empty else 0
            rev_impact = ((rev_after - rev_before) / rev_before) * 100 if rev_before > 0 else 0.0

            st.markdown(f"### 📊 Cannibalization Impact on **{incumbent_brand}**")
            st.caption(f"Comparing a {window_days}-day window before and after **{entrant_brand}** opened on {entry_date.strftime('%d %b %Y')}.")
            
            # Render Revenue Metrics
            st.markdown("**💰 Revenue Impact**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Pre-Entry Daily Avg", f"₹ {rev_before:,.0f}")
            m2.metric("Post-Entry Daily Avg", f"₹ {rev_after:,.0f}")
            m3.metric("Overall Revenue Impact", f"{rev_impact:+.1f}%", delta_color="inverse")

            # Render Advanced Invoice Metrics (Only if Invoice Data is present)
            if has_invoices:
                vol_before = before_df['Daily_Volume'].mean() if not before_df.empty else 0
                vol_after = after_df['Daily_Volume'].mean() if not after_df.empty else 0
                vol_impact = ((vol_after - vol_before) / vol_before) * 100 if vol_before > 0 else 0.0

                atv_before = before_df['Daily_ATV'].mean() if not before_df.empty else 0
                atv_after = after_df['Daily_ATV'].mean() if not after_df.empty else 0
                atv_impact = ((atv_after - atv_before) / atv_before) * 100 if atv_before > 0 else 0.0

                st.divider()
                st.markdown("**🛒 Behavioral Breakdown (Invoice Data)**")
                
                col_v1, col_v2, col_v3, col_a1, col_a2, col_a3 = st.columns(6)
                
                with col_v1:
                    st.metric("Pre-Entry Vol", f"{vol_before:,.0f} inv/day")
                with col_v2:
                    st.metric("Post-Entry Vol", f"{vol_after:,.0f} inv/day")
                with col_v3:
                    st.metric("Footfall Impact", f"{vol_impact:+.1f}%", delta_color="inverse")
                    
                with col_a1:
                    st.metric("Pre-Entry ATV", f"₹ {atv_before:,.0f}")
                with col_a2:
                    st.metric("Post-Entry ATV", f"₹ {atv_after:,.0f}")
                with col_a3:
                    st.metric("Ticket Size Impact", f"{atv_impact:+.1f}%", delta_color="inverse")

                # Behavioral Insight Logic
                if vol_impact < -5 and atv_impact > -5:
                    st.info("🧠 **Insight:** The competitor is stealing footfall. Customers are visiting less often, but when they do, they spend the same amount.")
                elif atv_impact < -5 and vol_impact > -5:
                    st.info("🧠 **Insight:** The competitor is stealing wallet share. Customers are still visiting you just as often, but they are spending less per visit.")
                elif vol_impact < -5 and atv_impact < -5:
                    st.error("🚨 **Insight:** Severe Cannibalization. The competitor is stealing both your footfall and causing remaining customers to spend less.")

            # --- PLOTLY CHART ---
            st.divider()
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=incumbent_df['Date'], 
                y=incumbent_df['7D_Rolling_Avg'],
                mode='lines',
                name=f"{incumbent_brand} (7D Avg)",
                line=dict(color='#3b82f6', width=3, shape='spline'),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)',
                hovertemplate='<b>%{x|%d %b %Y}</b><br>Sales Avg: ₹%{y:,.0f}<extra></extra>'
            ))
            
            fig.add_vline(x=entry_date.timestamp() * 1000, line_width=2, line_dash="dash", line_color="#ef4444")
            
            fig.add_annotation(
                x=entry_date.timestamp() * 1000,
                y=incumbent_df['7D_Rolling_Avg'].max(),
                text=f"🚨 {entrant_brand} Opens",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#ef4444",
                ax=-50,
                ay=-40,
                font=dict(color="white", size=12),
                bgcolor="#ef4444",
                borderpad=4
            )

            fig.update_layout(
                title=dict(text=f"Revenue Trend: {incumbent_brand}", font=dict(size=18)),
                xaxis_title="",
                yaxis_title="Daily Sales (₹)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified",
                xaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='#e5e7eb'),
                yaxis=dict(showgrid=True, gridcolor='#f3f4f6', zeroline=False),
                margin=dict(l=0, r=0, t=50, b=0),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            # ==========================================
            # HISTORICAL IMPACT: FUTURE FORECASTER
            # ==========================================
            with st.expander("🔮 Forecast Future Sales (Factoring in Competitor Impact)", expanded=False):
                st.markdown(f"Generate a future forecast for **{incumbent_brand}** that permanently bakes in the **{rev_impact:+.1f}%** revenue shift caused by **{entrant_brand}**.")
                
                col_hf1, col_hf2, col_hf3 = st.columns(3)
                with col_hf1:
                    h_target_month = st.number_input("Target Month", min_value=1, max_value=12, value=datetime.date.today().month, key="h_month")
                with col_hf2:
                    h_target_year = st.number_input("Target Year", min_value=2020, max_value=2100, value=datetime.date.today().year + 1, key="h_year")
                with col_hf3:
                    h_pred_type = st.selectbox("Forecast Type", ["Full Month", "Weekends Only", "Weekdays Only", "Custom Dates"], key="h_type")
                    
                h_custom_dates = ""
                if h_pred_type == "Custom Dates":
                    h_custom_dates = st.text_input("Enter dates (e.g., 1, 15, 25):", "1, 15", key="h_custom")

                if st.button("🚀 Run Impact-Adjusted Forecast", key="h_run"):
                    with st.spinner("Training model and applying impact multiplier..."):
                        # Prepare data & train model on incumbent's full history
                        incumbent_full = mall_df[mall_df[brand_col] == incumbent_brand].groupby('Date')[sales_col].sum().reset_index()
                        incumbent_full['Day_Number'] = incumbent_full['Date'].dt.dayofweek
                        incumbent_full['Day_of_Month'] = incumbent_full['Date'].dt.day
                        incumbent_full['Is_Weekend'] = incumbent_full['Day_Number'].isin([5, 6]).astype(int)
                        incumbent_full['Month'] = incumbent_full['Date'].dt.month
                        incumbent_full['Year'] = incumbent_full['Date'].dt.year
                        
                        features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(incumbent_full[features], incumbent_full[sales_col])

                        # Generate Future Dates
                        start_date = f"{h_target_year}-{h_target_month:02d}-01"
                        end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
                        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                        
                        h_pred_df = pd.DataFrame({'Date': future_dates})
                        h_pred_df['Day_Number'] = h_pred_df['Date'].dt.dayofweek
                        h_pred_df['Day_of_Month'] = h_pred_df['Date'].dt.day
                        h_pred_df['Is_Weekend'] = h_pred_df['Day_Number'].isin([5, 6]).astype(int)
                        h_pred_df['Month'] = h_pred_df['Date'].dt.month
                        h_pred_df['Year'] = h_pred_df['Date'].dt.year

                        # Filter based on type
                        if h_pred_type == "Weekends Only":
                            h_pred_df = h_pred_df[h_pred_df['Is_Weekend'] == 1]
                        elif h_pred_type == "Weekdays Only":
                            h_pred_df = h_pred_df[h_pred_df['Is_Weekend'] == 0]
                        elif h_pred_type == "Custom Dates" and h_custom_dates:
                            req_days = [int(d.strip()) for d in h_custom_dates.split(',')]
                            h_pred_df = h_pred_df[h_pred_df['Day_of_Month'].isin(req_days)]

                        # Predict Baseline AND apply the historical impact
                        h_pred_df['Base_Forecast'] = model.predict(h_pred_df[features])
                        h_pred_df['Adjusted_Forecast'] = h_pred_df['Base_Forecast'] * (1 + (rev_impact / 100))
                        
                        total_adj = h_pred_df['Adjusted_Forecast'].sum()
                        total_base = h_pred_df['Base_Forecast'].sum()
                        diff = total_adj - total_base
                        
                        # Output
                        st.success("Forecast generated!")
                        st.metric(f"Total Projected Revenue ({h_pred_type})", f"₹ {total_adj:,.0f}", f"{diff:+,.0f} vs Baseline", delta_color="inverse" if rev_impact < 0 else "normal")
                        
                        # Graph Base vs Adjusted
                        h_fig = go.Figure()
                        h_fig.add_trace(go.Bar(x=h_pred_df['Day_of_Month'], y=h_pred_df['Base_Forecast'], name="Without Competitor", marker_color='lightgray'))
                        h_fig.add_trace(go.Bar(x=h_pred_df['Day_of_Month'], y=h_pred_df['Adjusted_Forecast'], name="With Competitor Impact", marker_color='#ef4444' if rev_impact < 0 else '#10b981'))
                        h_fig.update_layout(title="Baseline Forecast vs. Impact-Adjusted Forecast", barmode='group', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Day of Month", yaxis_title="Projected Sales (₹)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(h_fig, use_container_width=True, config={'displayModeBar': False})


        # ==========================================
        # MANUAL IMPACT SIMULATION: MARKET SHARE ATTRITION
        # ==========================================


        st.divider()
        st.header("🔮 Manual Simulation: Market Share Attrition")
        st.markdown("Simulate a hypothetical new competitor entering the market and visualize how they dilute the market share of your top brands.")

        # --- 1. SIMULATION INPUTS ---
        col_s1, col_s2, col_s3 = st.columns([1, 1.5, 1])

        with col_s1:
            # Multi-select for Malls
            malls_sim = sorted(list(combined_df[mall_col].dropna().unique()))
            selected_malls_sim = st.multiselect("🏢 Select Mall(s)", malls_sim, default=malls_sim[:1], key="sim_malls")

        with col_s2:
            # Filter brands based on selected malls and calculate their total historic revenue
            sim_mall_df = combined_df[combined_df[mall_col].isin(selected_malls_sim)]
            brand_totals = sim_mall_df.groupby(brand_col)[sales_col].sum().reset_index()
            brand_totals = brand_totals.sort_values(by=sales_col, ascending=False)
            top_brands_list = brand_totals[brand_col].tolist()
            
            # Multi-select for Victim Brands (Default to the top 3 highest earners)
            default_victims = top_brands_list[:3] if len(top_brands_list) >= 3 else top_brands_list
            selected_victims = st.multiselect(
                "🎯 Select 'Victim' Brands", 
                top_brands_list, 
                default=default_victims,
                key="sim_victims",
                help="These are the incumbent brands that will lose market share to the new entrant."
            )

        with col_s3:
            # Slider for Market Share Attrition
            attrition_rate = st.slider(
                "📉 Share Captured by Entrant (%)", 
                min_value=1, max_value=50, value=15, step=1,
                help="Percentage of the selected brands' customers that will migrate to the new competitor."
            )

        # --- 2. SIMULATION LOGIC & MATH ---
        if selected_victims:
            # Filter down to just the selected victim brands
            victim_df = brand_totals[brand_totals[brand_col].isin(selected_victims)].copy()
            base_total = victim_df[sales_col].sum()
            
            # Apply the attrition math
            victim_df['Projected_Sales'] = victim_df[sales_col] * (1 - (attrition_rate / 100))
            new_entrant_sales = base_total * (attrition_rate / 100)
            
            # --- 3. METRICS OUTPUT ---
            st.markdown("### 💸 Projected Financial Impact")
            m1, m2, m3 = st.columns(3)
            m1.metric("Base Combined Revenue", f"₹ {base_total:,.0f}")
            m2.metric("New Competitor Revenue", f"₹ {new_entrant_sales:,.0f}", f"+{attrition_rate}% captured")
            m3.metric("Victims Retained Revenue", f"₹ {victim_df['Projected_Sales'].sum():,.0f}", f"-{attrition_rate}% lost", delta_color="inverse")
            
            # --- 4. PLOTLY DONUT CHARTS (BEFORE & AFTER) ---
            from plotly.subplots import make_subplots
            
            fig2 = make_subplots(
                rows=1, cols=2, 
                specs=[[{'type': 'domain'}, {'type': 'domain'}]], 
                subplot_titles=['<b>Before</b> (Incumbents Only)', '<b>After</b> (New Entrant Arrives)']
            )
            
            # Chart 1: Before Pie
            fig2.add_trace(go.Pie(
                labels=victim_df[brand_col], 
                values=victim_df[sales_col], 
                hole=0.45,
                name="Before",
                marker=dict(line=dict(color='#ffffff', width=2)),
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>"
            ), 1, 1)
            
            # Chart 2: After Pie (Adding the New Entrant)
            after_labels = victim_df[brand_col].tolist() + ['🚨 NEW COMPETITOR']
            after_values = victim_df['Projected_Sales'].tolist() + [new_entrant_sales]
            
            # Pull the new competitor's slice out slightly for visual emphasis
            pull_array = [0] * len(victim_df) + [0.1] 
            
            fig2.add_trace(go.Pie(
                labels=after_labels, 
                values=after_values, 
                hole=0.45,
                name="After",
                pull=pull_array,
                marker=dict(line=dict(color='#ffffff', width=2)),
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>"
            ), 1, 2)
            
            # Professional UI Layout
            fig2.update_layout(
                margin=dict(t=60, b=20, l=0, r=0),
                legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

            # ==========================================
            # MANUAL IMPACT: FUTURE FORECASTER
            # ==========================================
            with st.expander("🔮 Forecast Future Sales (Using Manual Attrition)", expanded=False):
                st.markdown(f"Project future sales for your selected 'Victim Brands' factoring in the **-{attrition_rate}%** loss to the new competitor.")
                
                col_mf1, col_mf2, col_mf3 = st.columns(3)
                with col_mf1:
                    m_target_month = st.number_input("Target Month", min_value=1, max_value=12, value=datetime.date.today().month, key="m_month")
                with col_mf2:
                    m_target_year = st.number_input("Target Year", min_value=2020, max_value=2100, value=datetime.date.today().year + 1, key="m_year")
                with col_mf3:
                    m_pred_type = st.selectbox("Forecast Type", ["Full Month", "Weekends Only", "Weekdays Only", "Custom Dates"], key="m_type")
                    
                m_custom_dates = ""
                if m_pred_type == "Custom Dates":
                    m_custom_dates = st.text_input("Enter dates (e.g., 1, 15, 25):", "1, 15", key="m_custom")

                if st.button("🚀 Run Simulation Forecast", key="m_run"):
                    with st.spinner("Aggregating victim brands and running model..."):
                        # Group all victim brands together for a combined forecast
                        victims_history = sim_mall_df[sim_mall_df[brand_col].isin(selected_victims)].groupby('Date')[sales_col].sum().reset_index()
                        victims_history['Day_Number'] = victims_history['Date'].dt.dayofweek
                        victims_history['Day_of_Month'] = victims_history['Date'].dt.day
                        victims_history['Is_Weekend'] = victims_history['Day_Number'].isin([5, 6]).astype(int)
                        victims_history['Month'] = victims_history['Date'].dt.month
                        victims_history['Year'] = victims_history['Date'].dt.year
                        
                        features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
                        m_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        m_model.fit(victims_history[features], victims_history[sales_col])

                        # Generate Future Dates
                        start_date = f"{m_target_year}-{m_target_month:02d}-01"
                        end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
                        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                        
                        m_pred_df = pd.DataFrame({'Date': future_dates})
                        m_pred_df['Day_Number'] = m_pred_df['Date'].dt.dayofweek
                        m_pred_df['Day_of_Month'] = m_pred_df['Date'].dt.day
                        m_pred_df['Is_Weekend'] = m_pred_df['Day_Number'].isin([5, 6]).astype(int)
                        m_pred_df['Month'] = m_pred_df['Date'].dt.month
                        m_pred_df['Year'] = m_pred_df['Date'].dt.year

                        # Filter based on type
                        if m_pred_type == "Weekends Only":
                            m_pred_df = m_pred_df[m_pred_df['Is_Weekend'] == 1]
                        elif m_pred_type == "Weekdays Only":
                            m_pred_df = m_pred_df[m_pred_df['Is_Weekend'] == 0]
                        elif m_pred_type == "Custom Dates" and m_custom_dates:
                            m_req_days = [int(d.strip()) for d in m_custom_dates.split(',')]
                            m_pred_df = m_pred_df[m_pred_df['Day_of_Month'].isin(m_req_days)]

                        # Predict Baseline AND apply the manual attrition rate
                        m_pred_df['Base_Forecast'] = m_model.predict(m_pred_df[features])
                        m_pred_df['Diluted_Forecast'] = m_pred_df['Base_Forecast'] * (1 - (attrition_rate / 100))
                        
                        m_total_adj = m_pred_df['Diluted_Forecast'].sum()
                        m_total_base = m_pred_df['Base_Forecast'].sum()
                        m_diff = m_total_adj - m_total_base
                        
                        # Output
                        st.success("Forecast generated!")
                        st.metric(f"Total Projected Revenue for Victims ({m_pred_type})", f"₹ {m_total_adj:,.0f}", f"{m_diff:+,.0f} lost to competitor", delta_color="inverse")
                        
                        # Graph Area Chart
                        m_fig = go.Figure()
                        m_fig.add_trace(go.Scatter(x=m_pred_df['Day_of_Month'], y=m_pred_df['Base_Forecast'], name="Baseline (No Competitor)", mode='lines', line=dict(color='lightgray', dash='dash')))
                        m_fig.add_trace(go.Scatter(x=m_pred_df['Day_of_Month'], y=m_pred_df['Diluted_Forecast'], name="Diluted by Competitor", mode='lines', fill='tozeroy', line=dict(color='#f97316')))
                        m_fig.update_layout(title="Future Sales Attrition Curve", plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Day of Month", yaxis_title="Combined Sales (₹)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(m_fig, use_container_width=True, config={'displayModeBar': False})


# !***! !***! !***! !***! !***! !***! !***! !***! !***!

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