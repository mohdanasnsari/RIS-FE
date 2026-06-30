import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

class ForecastingTab:
    def __init__(self, dataframe):
        # Initialize the class with the uploaded data
        self.combined_df = dataframe

    def render(self):
        st.info("Forecasting Model - Data Prep & Aggregation")
        
        # --- PRE-CLEANING ---
        # Create a working copy so we don't modify the core dataframe used by other tabs
        working_df = self.combined_df.copy()
        working_df.columns = working_df.columns.str.strip()
        
        # --- 0. BULLETPROOF COLUMN MAPPING ---
        if 'Total Gross Amount' in working_df.columns:
            mall_col, brand_col, sales_col = 'Mall', 'Brand Name', 'Total Gross Amount'
        elif 'Gross Amount' in working_df.columns:
            mall_col, brand_col, sales_col = 'Mall Name', 'Tenant Name', 'Gross Amount'
        else:
            st.error("🚨 Unrecognized data format. Could not find 'Total Gross Amount' or 'Gross Amount'. Please clear data and re-upload.")
            st.stop()

        st.caption(f"⚙️ Auto-detected target metric: **{sales_col}**")

        # --- 1. SILENT DATA CLEANING & FEATURE ENGINEERING ---
        with st.spinner("Cleaning data and extracting time features..."):
            initial_rows = len(working_df)

            # Cleaning
            working_df = working_df.drop_duplicates()
            # Handle potential commas in string amounts before converting to numeric
            working_df[sales_col] = pd.to_numeric(working_df[sales_col].astype(str).str.replace(',', ''), errors='coerce')
            working_df['Date'] = pd.to_datetime(working_df['Date'], format='mixed', dayfirst=True, errors='coerce')
            
            # Drop invalid rows based on the dynamic sales column
            working_df = working_df.dropna(subset=['Date', sales_col])
            
            # Feature Extraction
            working_df['Day_Name'] = working_df['Date'].dt.day_name()
            working_df['Day_Number'] = working_df['Date'].dt.dayofweek
            working_df['Is_Weekend'] = working_df['Day_Number'].isin([5, 6]).astype(int)
            working_df['Month'] = working_df['Date'].dt.month
            working_df['Year'] = working_df['Date'].dt.year
            working_df['Day_of_Month'] = working_df['Date'].dt.day

        # --- 2. ESSENTIAL METRICS ONLY ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Valid Transactions", f"{len(working_df):,}", f"{initial_rows - len(working_df)} invalid dropped", delta_color="off")
        col2.metric("Unique Dates", working_df['Date'].nunique())
        col3.metric("Date Range", f"{working_df['Date'].dt.date.min()} to {working_df['Date'].dt.date.max()}")

        st.divider()

        # --- 3. DYNAMIC CASCADING FILTERING ---
        st.subheader("Scope Selection")
        
        col_f1, col_f2 = st.columns(2)
        
        # Mall Filter
        with col_f1:
            if mall_col in working_df.columns:
                malls = ["All"] + sorted(list(working_df[mall_col].dropna().unique()))
                selected_malls = st.multiselect("🏢 Filter by Mall(s)", malls, default=["All"])
            else:
                selected_malls = ["All"]
                st.warning("No Mall column found in this dataset.")

        # Cascading Logic: Filter dataset by selected malls *before* populating Brands
        temp_filtered_df = working_df.copy()
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