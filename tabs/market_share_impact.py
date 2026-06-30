import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from plotly.subplots import make_subplots

class MarketImpactTab:
    def __init__(self, dataframe):
        # Initialize the class with the uploaded data
        self.combined_df = dataframe

    def render(self):
        st.header("Market Share Impact")
        st.markdown("Analyze historical cannibalization or simulate future market shifts.")

        # --- PRE-CLEANING ---
        working_df = self.combined_df.copy()
        working_df.columns = working_df.columns.str.strip()

        # --- 0. DYNAMIC COLUMN MAPPING ---
        if 'Total Gross Amount' in working_df.columns:
            mall_col, brand_col, sales_col = 'Mall', 'Brand Name', 'Total Gross Amount'
        elif 'Gross Amount' in working_df.columns:
            mall_col, brand_col, sales_col = 'Mall Name', 'Tenant Name', 'Gross Amount'
        else:
            st.error("🚨 Unrecognized data format. Could not find 'Total Gross Amount' or 'Gross Amount'.")
            st.stop()

        # ==========================================
        # SCENARIO SUB-TABS
        # ==========================================
        sub_tabs = st.tabs(["⚔️ Scenario 1: Competitor Entry Impact", "📈 Scenario 2: Price Elasticity (Coming Soon)"])

        # ---------------------------------------------------------
        # SUB-TAB 1: COMPETITOR IMPACT (Historical + Manual)
        # ---------------------------------------------------------
        with sub_tabs[0]:
            
            st.subheader("Historical Competitor Impact")
            st.markdown("Use actual historical data to see how the arrival of a new brand impacted existing competitors.")

            # --- 1. MALL SELECTION ---
            malls = sorted(list(working_df[mall_col].dropna().unique()))
            selected_mall = st.selectbox("🏢 Select Mall for Analysis", malls, key="impact_mall_select")
            
            mall_df = working_df[working_df[mall_col] == selected_mall].copy()

            # --- 2. BRAND ENTRY TIMELINE ---
            arrival_dates = mall_df.groupby(brand_col)['Date'].min().reset_index()
            arrival_dates.columns = ['Brand Name', 'First Arrival Date']
            arrival_dates = arrival_dates.sort_values('First Arrival Date').reset_index(drop=True)

            st.divider()
            st.markdown(f"**Brand Entry Timeline: {selected_mall}**")
            
            col_list, col_selectors = st.columns([1.2, 2])
            
            with col_list:
                st.caption("Review arrival dates to match competitors.")
                display_dates = arrival_dates.copy()
                display_dates['First Arrival Date'] = display_dates['First Arrival Date'].dt.strftime('%d %b %Y')
                st.dataframe(display_dates, use_container_width=True, hide_index=True, height=250)

            # --- 3. THE PRE/POST SELECTION ---
            with col_selectors:
                st.caption("Select your comparison targets.")
                incumbent_brand = st.selectbox("🛡️ Select Incumbent Brand (The older brand)", arrival_dates['Brand Name'].tolist())
                entrant_brand = st.selectbox("⚔️ Select New Entrant (The competitor)", arrival_dates['Brand Name'].tolist())
                
                entry_date = arrival_dates.loc[arrival_dates['Brand Name'] == entrant_brand, 'First Arrival Date'].values[0]
                entry_date = pd.to_datetime(entry_date)
                
                incumbent_start = arrival_dates.loc[arrival_dates['Brand Name'] == incumbent_brand, 'First Arrival Date'].values[0]
                
                if pd.to_datetime(incumbent_start) >= entry_date and incumbent_brand != entrant_brand:
                    st.warning("⚠️ The Incumbent brand arrived ON or AFTER the Entrant brand. Please select an older incumbent to see a valid 'Before & After' impact.")

            # --- 4. THE ANALYSIS ENGINE ---
            if incumbent_brand and entrant_brand and incumbent_brand != entrant_brand and pd.to_datetime(incumbent_start) < entry_date:
                st.divider()
                
                has_invoices = 'Invoice Number' in mall_df.columns
                
                if has_invoices:
                    incumbent_df = mall_df[mall_df[brand_col] == incumbent_brand].groupby('Date').agg(
                        Daily_Sales=(sales_col, 'sum'),
                        Daily_Volume=('Invoice Number', 'nunique')
                    ).reset_index()
                    incumbent_df['Daily_ATV'] = incumbent_df['Daily_Sales'] / incumbent_df['Daily_Volume']
                else:
                    incumbent_df = mall_df[mall_df[brand_col] == incumbent_brand].groupby('Date').agg(
                        Daily_Sales=(sales_col, 'sum')
                    ).reset_index()

                incumbent_df = incumbent_df.sort_values('Date')
                incumbent_df['7D_Rolling_Avg'] = incumbent_df['Daily_Sales'].rolling(window=7, min_periods=1).mean()
                
                window_days = 45
                before_df = incumbent_df[(incumbent_df['Date'] >= (entry_date - pd.Timedelta(days=window_days))) & (incumbent_df['Date'] < entry_date)]
                after_df = incumbent_df[(incumbent_df['Date'] >= entry_date) & (incumbent_df['Date'] <= (entry_date + pd.Timedelta(days=window_days)))]
                
                rev_before = before_df['Daily_Sales'].mean() if not before_df.empty else 0
                rev_after = after_df['Daily_Sales'].mean() if not after_df.empty else 0
                rev_impact = ((rev_after - rev_before) / rev_before) * 100 if rev_before > 0 else 0.0

                st.markdown(f"### 📊 Cannibalization Impact on **{incumbent_brand}**")
                st.caption(f"Comparing a {window_days}-day window before and after **{entrant_brand}** opened on {entry_date.strftime('%d %b %Y')}.")
                
                st.markdown("**💰 Revenue Impact**")
                m1, m2, m3 = st.columns(3)
                m1.metric("Pre-Entry Daily Avg", f"₹ {rev_before:,.0f}")
                m2.metric("Post-Entry Daily Avg", f"₹ {rev_after:,.0f}")
                m3.metric("Overall Revenue Impact", f"{rev_impact:+.1f}%", delta_color="inverse")

                mall_daily = mall_df.groupby('Date')[sales_col].sum().reset_index()
                mall_before = mall_daily[(mall_daily['Date'] >= (entry_date - pd.Timedelta(days=window_days))) & (mall_daily['Date'] < entry_date)]
                mall_after = mall_daily[(mall_daily['Date'] >= entry_date) & (mall_daily['Date'] <= (entry_date + pd.Timedelta(days=window_days)))]
                
                mall_rev_before = mall_before[sales_col].mean() if not mall_before.empty else 0
                mall_rev_after = mall_after[sales_col].mean() if not mall_after.empty else 0
                mall_impact = ((mall_rev_after - mall_rev_before) / mall_rev_before) * 100 if mall_rev_before > 0 else 0.0

                yoy_start = entry_date - pd.DateOffset(years=1)
                yoy_end = (entry_date + pd.Timedelta(days=window_days)) - pd.DateOffset(years=1)
                yoy_df = incumbent_df[(incumbent_df['Date'] >= yoy_start) & (incumbent_df['Date'] <= yoy_end)].copy()
                
                yoy_avg = yoy_df['Daily_Sales'].mean() if not yoy_df.empty else 0
                yoy_impact = ((rev_after - yoy_avg) / yoy_avg) * 100 if yoy_avg > 0 else None

                st.divider()
                st.markdown("**🌍 Macro-Environmental Checks (Defeating Inflation & Seasonality)**")
                st.caption("Compare the incumbent's performance against the overall mall trend and their own historical Year-over-Year baseline.")
                
                c_e1, c_e2, c_e3 = st.columns(3)
                c_e1.metric("Incumbent Revenue Shift", f"{rev_impact:+.1f}%")
                c_e2.metric(f"Overall {selected_mall} Shift", f"{mall_impact:+.1f}%")
                
                if yoy_impact is not None:
                    c_e3.metric("Incumbent YoY Shift (Seasonality)", f"{yoy_impact:+.1f}%", help=f"Compared to their daily average of ₹{yoy_avg:,.0f} during {yoy_start.strftime('%b %Y')}")
                else:
                    c_e3.metric("Incumbent YoY Shift", "N/A", help="Not enough historical data from the previous year.")
                    
                if rev_impact > 0 and mall_impact > rev_impact:
                    st.warning(f"📉 **Hidden Loss:** {incumbent_brand} grew by {rev_impact:.1f}%, but the overall mall grew by {mall_impact:.1f}%. The incumbent actually **lost relative market share** despite making more money (likely due to inflation or seasonal spikes).")
                elif rev_impact < 0 and mall_impact < rev_impact:
                    st.success(f"🛡️ **Resilience:** {incumbent_brand} dropped by {abs(rev_impact):.1f}%, but the overall mall dropped worse ({abs(mall_impact):.1f}%). The incumbent is actually **outperforming the market** despite the new competitor.")
                elif rev_impact > 0:
                    st.success(f"📈 **True Growth:** {incumbent_brand} is growing faster than the mall average. The new competitor may have created a 'Halo Effect' drawing more footfall to this specific category.")
                else:
                    st.error(f"🚨 **True Cannibalization:** The incumbent is shrinking while the mall is not. {entrant_brand} is directly stealing market share.")

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

                    if vol_impact < -5 and atv_impact > -5:
                        st.info("🧠 **Insight:** The competitor is stealing footfall. Customers are visiting less often, but when they do, they spend the same amount.")
                    elif atv_impact < -5 and vol_impact > -5:
                        st.info("🧠 **Insight:** The competitor is stealing wallet share. Customers are still visiting you just as often, but they are spending less per visit.")
                    elif vol_impact < -5 and atv_impact < -5:
                        st.error("🚨 **Insight:** Severe Cannibalization. The competitor is stealing both your footfall and causing remaining customers to spend less.")
                else:
                    st.divider()
                    st.markdown(f"**📅 Year-Over-Year Seasonality Trend: {incumbent_brand}**")
                    st.caption(f"Comparing {incumbent_brand}'s sales following the competitor entry ({entry_date.strftime('%b %Y')}) against the exact same calendar period from the previous year ({yoy_start.strftime('%b %Y')}).")
                    
                    if not yoy_df.empty and not after_df.empty:
                        yoy_plot = yoy_df.copy().reset_index(drop=True)
                        after_plot = after_df.copy().reset_index(drop=True)
                        
                        yoy_plot['Day_Sequence'] = range(1, len(yoy_plot) + 1)
                        after_plot['Day_Sequence'] = range(1, len(after_plot) + 1)
                        
                        if '7D_Rolling_Avg' not in yoy_plot.columns:
                            yoy_plot['7D_Rolling_Avg'] = yoy_plot['Daily_Sales'].rolling(window=7, min_periods=1).mean()
                        
                        fig_yoy = go.Figure()
                        
                        fig_yoy.add_trace(go.Scatter(
                            x=yoy_plot['Day_Sequence'], 
                            y=yoy_plot['7D_Rolling_Avg'],
                            mode='lines',
                            name=f"Previous Year ({yoy_start.strftime('%b %Y')})",
                            line=dict(color='lightgray', width=2, dash='dash'),
                            hovertemplate='<b>Day %{x}</b><br>Past Year: ₹%{y:,.0f}<extra></extra>'
                        ))
                        
                        fig_yoy.add_trace(go.Scatter(
                            x=after_plot['Day_Sequence'], 
                            y=after_plot['7D_Rolling_Avg'],
                            mode='lines',
                            name=f"Current Year ({entry_date.strftime('%b %Y')})",
                            line=dict(color='#f97316', width=3, shape='spline'),
                            fill='tozeroy',
                            fillcolor='rgba(249, 115, 22, 0.1)',
                            hovertemplate='<b>Day %{x}</b><br>Current Year: ₹%{y:,.0f}<extra></extra>'
                        ))
                        
                        fig_yoy.update_layout(
                            title=f"Seasonality Check: {incumbent_brand}", 
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)',
                            xaxis_title="Days Post-Entry (1 to 45)", 
                            yaxis_title="7-Day Rolling Sales (₹)", 
                            hovermode="x unified",
                            xaxis=dict(showgrid=False, showline=True, linewidth=1, linecolor='#e5e7eb'),
                            yaxis=dict(showgrid=True, gridcolor='#f3f4f6', zeroline=False),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_yoy, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("Insufficient historical data from the previous year to render a Year-Over-Year seasonality chart.")

                # --- MAIN PLOTLY CHART ---
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

                # --- HISTORICAL FORECASTER ---
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
                            incumbent_full = mall_df[mall_df[brand_col] == incumbent_brand].groupby('Date')[sales_col].sum().reset_index()
                            incumbent_full['Day_Number'] = incumbent_full['Date'].dt.dayofweek
                            incumbent_full['Day_of_Month'] = incumbent_full['Date'].dt.day
                            incumbent_full['Is_Weekend'] = incumbent_full['Day_Number'].isin([5, 6]).astype(int)
                            incumbent_full['Month'] = incumbent_full['Date'].dt.month
                            incumbent_full['Year'] = incumbent_full['Date'].dt.year
                            
                            features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(incumbent_full[features], incumbent_full[sales_col])

                            start_date = f"{h_target_year}-{h_target_month:02d}-01"
                            end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
                            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                            
                            h_pred_df = pd.DataFrame({'Date': future_dates})
                            h_pred_df['Day_Number'] = h_pred_df['Date'].dt.dayofweek
                            h_pred_df['Day_of_Month'] = h_pred_df['Date'].dt.day
                            h_pred_df['Is_Weekend'] = h_pred_df['Day_Number'].isin([5, 6]).astype(int)
                            h_pred_df['Month'] = h_pred_df['Date'].dt.month
                            h_pred_df['Year'] = h_pred_df['Date'].dt.year

                            if h_pred_type == "Weekends Only":
                                h_pred_df = h_pred_df[h_pred_df['Is_Weekend'] == 1]
                            elif h_pred_type == "Weekdays Only":
                                h_pred_df = h_pred_df[h_pred_df['Is_Weekend'] == 0]
                            elif h_pred_type == "Custom Dates" and h_custom_dates:
                                req_days = [int(d.strip()) for d in h_custom_dates.split(',')]
                                h_pred_df = h_pred_df[h_pred_df['Day_of_Month'].isin(req_days)]

                            h_pred_df['Base_Forecast'] = model.predict(h_pred_df[features])
                            h_pred_df['Adjusted_Forecast'] = h_pred_df['Base_Forecast'] * (1 + (rev_impact / 100))
                            
                            total_adj = h_pred_df['Adjusted_Forecast'].sum()
                            total_base = h_pred_df['Base_Forecast'].sum()
                            diff = total_adj - total_base
                            
                            st.success("Forecast generated!")
                            st.metric(f"Total Projected Revenue ({h_pred_type})", f"₹ {total_adj:,.0f}", f"{diff:+,.0f} vs Baseline", delta_color="inverse" if rev_impact < 0 else "normal")
                            
                            h_fig = go.Figure()
                            h_fig.add_trace(go.Bar(x=h_pred_df['Day_of_Month'], y=h_pred_df['Base_Forecast'], name="Without Competitor", marker_color='lightgray'))
                            h_fig.add_trace(go.Bar(x=h_pred_df['Day_of_Month'], y=h_pred_df['Adjusted_Forecast'], name="With Competitor Impact", marker_color='#ef4444' if rev_impact < 0 else '#10b981'))
                            h_fig.update_layout(title="Baseline Forecast vs. Impact-Adjusted Forecast", barmode='group', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Day of Month", yaxis_title="Projected Sales (₹)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            st.plotly_chart(h_fig, use_container_width=True, config={'displayModeBar': False})

            # ==========================================
            # MANUAL IMPACT SIMULATION: MARKET SHARE ATTRITION
            # ==========================================
            st.divider()
            st.subheader("Manual Simulation: Market Share Attrition")
            st.markdown("Simulate a hypothetical new competitor entering the market and visualize how they dilute the market share of your top brands.")

            # --- 1. SIMULATION INPUTS ---
            col_s1, col_s2, col_s3 = st.columns([1, 1.5, 1])

            with col_s1:
                malls_sim = sorted(list(working_df[mall_col].dropna().unique()))
                selected_malls_sim = st.multiselect("🏢 Select Mall(s)", malls_sim, default=malls_sim[:1], key="sim_malls")

            with col_s2:
                sim_mall_df = working_df[working_df[mall_col].isin(selected_malls_sim)]
                brand_totals = sim_mall_df.groupby(brand_col)[sales_col].sum().reset_index()
                brand_totals = brand_totals.sort_values(by=sales_col, ascending=False)
                top_brands_list = brand_totals[brand_col].tolist()
                
                default_victims = top_brands_list[:3] if len(top_brands_list) >= 3 else top_brands_list
                selected_victims = st.multiselect(
                    "🎯 Select 'Victim' Brands", 
                    top_brands_list, 
                    default=default_victims,
                    key="sim_victims",
                    help="These are the incumbent brands that will lose market share to the new entrant."
                )

            with col_s3:
                attrition_rate = st.slider(
                    "📉 Share Captured by Entrant (%)", 
                    min_value=1, max_value=50, value=15, step=1,
                    help="Percentage of the selected brands' customers that will migrate to the new competitor."
                )

            # --- 2. SIMULATION LOGIC & MATH ---
            if selected_victims:
                victim_df = brand_totals[brand_totals[brand_col].isin(selected_victims)].copy()
                base_total = victim_df[sales_col].sum()
                
                victim_df['Projected_Sales'] = victim_df[sales_col] * (1 - (attrition_rate / 100))
                new_entrant_sales = base_total * (attrition_rate / 100)
                
                # --- 3. METRICS OUTPUT ---
                st.markdown("### 💸 Projected Financial Impact")
                m1, m2, m3 = st.columns(3)
                m1.metric("Base Combined Revenue", f"₹ {base_total:,.0f}")
                m2.metric("New Competitor Revenue", f"₹ {new_entrant_sales:,.0f}", f"+{attrition_rate}% captured")
                m3.metric("Victims Retained Revenue", f"₹ {victim_df['Projected_Sales'].sum():,.0f}", f"-{attrition_rate}% lost", delta_color="inverse")
                
                # --- 4. PLOTLY DONUT CHARTS (BEFORE & AFTER) ---
                fig2 = make_subplots(
                    rows=1, cols=2, 
                    specs=[[{'type': 'domain'}, {'type': 'domain'}]], 
                    subplot_titles=['<b>Before</b> (Incumbents Only)', '<b>After</b> (New Entrant Arrives)']
                )
                
                fig2.add_trace(go.Pie(
                    labels=victim_df[brand_col], 
                    values=victim_df[sales_col], 
                    hole=0.45,
                    name="Before",
                    marker=dict(line=dict(color='#ffffff', width=2)),
                    hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>"
                ), 1, 1)
                
                after_labels = victim_df[brand_col].tolist() + ['🚨 NEW COMPETITOR']
                after_values = victim_df['Projected_Sales'].tolist() + [new_entrant_sales]
                
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
                
                fig2.update_layout(
                    margin=dict(t=60, b=20, l=0, r=0),
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

                # --- MANUAL FORECASTER ---
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
                            victims_history = sim_mall_df[sim_mall_df[brand_col].isin(selected_victims)].groupby('Date')[sales_col].sum().reset_index()
                            victims_history['Day_Number'] = victims_history['Date'].dt.dayofweek
                            victims_history['Day_of_Month'] = victims_history['Date'].dt.day
                            victims_history['Is_Weekend'] = victims_history['Day_Number'].isin([5, 6]).astype(int)
                            victims_history['Month'] = victims_history['Date'].dt.month
                            victims_history['Year'] = victims_history['Date'].dt.year
                            
                            features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
                            m_model = RandomForestRegressor(n_estimators=100, random_state=42)
                            m_model.fit(victims_history[features], victims_history[sales_col])

                            start_date = f"{m_target_year}-{m_target_month:02d}-01"
                            end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1)
                            future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                            
                            m_pred_df = pd.DataFrame({'Date': future_dates})
                            m_pred_df['Day_Number'] = m_pred_df['Date'].dt.dayofweek
                            m_pred_df['Day_of_Month'] = m_pred_df['Date'].dt.day
                            m_pred_df['Is_Weekend'] = m_pred_df['Day_Number'].isin([5, 6]).astype(int)
                            m_pred_df['Month'] = m_pred_df['Date'].dt.month
                            m_pred_df['Year'] = m_pred_df['Date'].dt.year

                            if m_pred_type == "Weekends Only":
                                m_pred_df = m_pred_df[m_pred_df['Is_Weekend'] == 1]
                            elif m_pred_type == "Weekdays Only":
                                m_pred_df = m_pred_df[m_pred_df['Is_Weekend'] == 0]
                            elif m_pred_type == "Custom Dates" and m_custom_dates:
                                m_req_days = [int(d.strip()) for d in m_custom_dates.split(',')]
                                m_pred_df = m_pred_df[m_pred_df['Day_of_Month'].isin(m_req_days)]

                            m_pred_df['Base_Forecast'] = m_model.predict(m_pred_df[features])
                            m_pred_df['Diluted_Forecast'] = m_pred_df['Base_Forecast'] * (1 - (attrition_rate / 100))
                            
                            m_total_adj = m_pred_df['Diluted_Forecast'].sum()
                            m_total_base = m_pred_df['Base_Forecast'].sum()
                            m_diff = m_total_adj - m_total_base
                            
                            st.success("Forecast generated!")
                            st.metric(f"Total Projected Revenue for Victims ({m_pred_type})", f"₹ {m_total_adj:,.0f}", f"{m_diff:+,.0f} lost to competitor", delta_color="inverse")
                            
                            m_fig = go.Figure()
                            m_fig.add_trace(go.Scatter(x=m_pred_df['Day_of_Month'], y=m_pred_df['Base_Forecast'], name="Baseline (No Competitor)", mode='lines', line=dict(color='lightgray', dash='dash')))
                            m_fig.add_trace(go.Scatter(x=m_pred_df['Day_of_Month'], y=m_pred_df['Diluted_Forecast'], name="Diluted by Competitor", mode='lines', fill='tozeroy', line=dict(color='#f97316')))
                            m_fig.update_layout(title="Future Sales Attrition Curve", plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Day of Month", yaxis_title="Combined Sales (₹)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                            st.plotly_chart(m_fig, use_container_width=True, config={'displayModeBar': False})

        # ---------------------------------------------------------
        # SUB-TAB 2: PRICE ELASTICITY (Coming Soon)
        # ---------------------------------------------------------
        with sub_tabs[1]:
            st.info("🚀 Price Elasticity Simulator is under construction and will be deployed here soon.")