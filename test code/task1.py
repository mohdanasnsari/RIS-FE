import pandas as pd
import glob
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

def merge_and_process_data():
    files = [
        f for f in glob.glob(r"E:\wovVRA Sales forecasting model\data\*.xlsx")
        if not f.split("\\")[-1].startswith("~$")
    ]

    print("Files found:", len(files))

    all_data = []

    for file in files:
        print("Reading:", file)
        df = pd.read_excel(file)
        all_data.append(df)

    print("Finished reading all files")

    merged_df = pd.concat(all_data, ignore_index = True)
    merged_df = merged_df.drop_duplicates()

    print("Rows:", len(merged_df))
    print("Columns:", len(merged_df.columns))

    # --------------------------------------- duplicates
    print("\nChecking duplicates...")
    duplicate_count = merged_df.duplicated().sum()
    print("Duplicate Rows:", duplicate_count)

    # --------------------------------------------- info
    print("\nColumn Information:", merged_df.info())   
    print("\nColumns:", merged_df.columns)

    # ----------------------------------------------- gross amount dtype
    print(merged_df['Total Gross Amount'].head(20))
    print(merged_df['Total Gross Amount'].sample(20))

    # ------------------------------------------------------ convert merge amount object to float64
    merged_df['Total Gross Amount'] = pd.to_numeric(
        merged_df['Total Gross Amount'],
        errors='coerce'
    )

    print(merged_df['Total Gross Amount'].dtype)

    merged_df = merged_df.dropna(subset=['Date', 'Total Gross Amount'])

    print("\nMissing Values:", merged_df.isnull().sum())  
    print(merged_df.shape)

    print("Earliest Date:", merged_df['Date'].min())   #checking earliest to latest date
    print("Latest Date:", merged_df['Date'].max())
    print("Unique Dates:", merged_df['Date'].nunique()) #unique dates count

    # day name
    merged_df['Day_Name'] = merged_df['Date'].dt.day_name()
    # day number
    merged_df['Day_Number'] = merged_df['Date'].dt.dayofweek
    # weekend flag
    merged_df['Is_Weekend'] = merged_df['Day_Number'].isin([5, 6])
    # month
    merged_df['Month'] = merged_df['Date'].dt.month
    # year
    merged_df['Year'] = merged_df['Date'].dt.year

    print(merged_df[['Date', 'Day_Name', 'Day_Number', 'Is_Weekend', 'Month', 'Year']].head())

    #Creating daily sales
    daily_sales = merged_df.groupby('Date')['Total Gross Amount'].sum().reset_index()

    print(daily_sales.head())
    print("Rows:", len(daily_sales))

    daily_sales['Day_Name'] = daily_sales['Date'].dt.day_name()
    daily_sales['Day_Number'] = daily_sales['Date'].dt.dayofweek
    daily_sales['Is_Weekend'] = daily_sales['Day_Number'].isin([5, 6])
    daily_sales['Month'] = daily_sales['Date'].dt.month
    daily_sales['Year'] = daily_sales['Date'].dt.year

    print(daily_sales.head())

    #---------------------------------------- saving the processed file
    merged_df.to_csv(
        r"E:\wovVRA Sales forecasting model\master_sales_clean.csv",
        index=False
    )

    daily_sales.to_csv(
        r"E:\wovVRA Sales forecasting model\daily_sales_clean.csv",
        index=False
    )

    print("Clean datasets saved successfully")
    
    return merged_df, daily_sales

# ----------------------------------------------- model training and other functions

def train_model(daily_sales_df):
    print("\nTraining Model...")  #model training on daily sales db
    
    daily_sales_df['Day_of_Month'] = daily_sales_df['Date'].dt.day  #feb 1 as 1, 2 as 2....
    
    features = ['Day_Number', 'Day_of_Month', 'Is_Weekend', 'Month', 'Year']
    
    X_train = daily_sales_df[features]
    y_train = daily_sales_df['Total Gross Amount']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model, features

# --------------------------------------- dates and feature for future prediction (kind of new db)

def generate_future_features(month, year):
    start_date = f"{year}-{month:02d}-01"  #generate full list of date 1,2,3,4....28,29,30,31
    end_date = pd.to_datetime(start_date) + pd.offsets.MonthEnd(1) #last date
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D') #sequence of dates
    
    prediction_df = pd.DataFrame({'Date': future_dates}) #new db for prediction instead of using daily sales
    #features
    prediction_df['Day_Name'] = prediction_df['Date'].dt.day_name()
    prediction_df['Day_Number'] = prediction_df['Date'].dt.dayofweek
    
    prediction_df['Day_of_Month'] = prediction_df['Date'].dt.day
    
    prediction_df['Is_Weekend'] = prediction_df['Day_Number'].isin([5, 6]).astype(int)
    prediction_df['Month'] = prediction_df['Date'].dt.month
    prediction_df['Year'] = prediction_df['Date'].dt.year
    
    return prediction_df, start_date

# --------------------------- matplotlib for graphs and all (need to work more on this! *paused for now)

# def plot_forecast(past_df, future_df, title, plot_type='line'):
#     has_past = not past_df.empty
    
#     past_total = 0
#     if has_past:
#         past_df = past_df.copy()
#         past_df['Day'] = past_df['Date'].dt.day
#         past_total = past_df['Total Gross Amount'].sum()
        
#     future_df = future_df.copy()
#     future_df['Day'] = future_df['Date'].dt.day
#     future_total = future_df['Predicted_Sales'].sum()

#     num_plots = 2 if has_past else 1
#     fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(14, 6 * num_plots), sharex=True)
    
#     if num_plots == 1:
#         axes = [axes]

#     box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')

#     if plot_type == 'line':
#         if has_past:
#             ax1 = axes[0]
#             ax1.plot(past_df['Day'], past_df['Total Gross Amount'], label='Actual (Last Year)', color='gray', marker='o', linestyle='dashed')
#             for x, y in zip(past_df['Day'], past_df['Total Gross Amount']):
#                 # Added rotation=90
#                 ax1.text(x, y + (past_df['Total Gross Amount'].max() * 0.02), f"{y:.2f}", ha='center', va='bottom', fontsize=8, color='dimgray', rotation=90)
            
#             ax1.set_title("Actual Data (Last Year)", fontsize=13, fontweight='bold', pad=10)
#             ax1.set_ylabel('Sales Amount', fontsize=12)
            
#             ax1.text(0.01, 0.95, f"Total Actual: {past_total:,.2f}", transform=ax1.transAxes, 
#                      fontsize=11, fontweight='bold', color='dimgray', va='top', bbox=box_props)
            
#             ymin, ymax = ax1.get_ylim()
#             ax1.set_ylim(ymin, ymax * 1.35) # Increased to 1.35 for vertical text headroom
#             ax1.legend(loc='upper right')
#             ax1.grid(True, alpha=0.3)
            
#             ax2 = axes[1]
#         else:
#             ax2 = axes[0]

#         ax2.plot(future_df['Day'], future_df['Predicted_Sales'], label='Predicted (Future)', color='blue', marker='o', linewidth=2)
#         for x, y in zip(future_df['Day'], future_df['Predicted_Sales']):
#             # Added rotation=90
#             ax2.text(x, y + (future_df['Predicted_Sales'].max() * 0.02), f"{y:.2f}", ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold', rotation=90)
            
#         ax2.set_title("Predicted Data (Future)", fontsize=13, fontweight='bold', pad=10)
#         ax2.set_ylabel('Sales Amount', fontsize=12)
        
#         ax2.text(0.01, 0.95, f"Total Predicted: {future_total:,.2f}", transform=ax2.transAxes, 
#                  fontsize=11, fontweight='bold', color='blue', va='top', bbox=box_props)
                 
#         ymin, ymax = ax2.get_ylim()
#         ax2.set_ylim(ymin, ymax * 1.35) # Increased to 1.35 for vertical text headroom
#         ax2.legend(loc='upper right')
#         ax2.grid(True, alpha=0.3)

#     elif plot_type == 'bar':
#         width = 0.5
#         if has_past:
#             ax1 = axes[0]
#             bars_past = ax1.bar(past_df['Day'], past_df['Total Gross Amount'], width=width, label='Actual (Last Year)', color='lightgray')
#             for bar in bars_past:
#                 yval = bar.get_height()
#                 # Added rotation=90
#                 ax1.text(bar.get_x() + bar.get_width()/2, yval + (past_df['Total Gross Amount'].max() * 0.02), f"{yval:.2f}", ha='center', va='bottom', fontsize=8, color='dimgray', rotation=90)
                
#             ax1.set_title("Actual Data (Last Year)", fontsize=13, fontweight='bold', pad=10)
#             ax1.set_ylabel('Sales Amount', fontsize=12)
            
#             ax1.text(0.01, 0.95, f"Total Actual: {past_total:,.2f}", transform=ax1.transAxes, 
#                      fontsize=11, fontweight='bold', color='dimgray', va='top', bbox=box_props)
                     
#             ymin, ymax = ax1.get_ylim()
#             ax1.set_ylim(0, ymax * 1.35) # Increased to 1.35 for vertical text headroom
#             ax1.legend(loc='upper right')
#             ax1.grid(True, axis='y', alpha=0.3)
            
#             ax2 = axes[1]
#         else:
#             ax2 = axes[0]

#         bars_future = ax2.bar(future_df['Day'], future_df['Predicted_Sales'], width=width, label='Predicted (Future)', color='blue')
#         for bar in bars_future:
#             yval = bar.get_height()
#             # Added rotation=90
#             ax2.text(bar.get_x() + bar.get_width()/2, yval + (future_df['Predicted_Sales'].max() * 0.02), f"{yval:.2f}", ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold', rotation=90)
            
#         ax2.set_title("Predicted Data (Future)", fontsize=13, fontweight='bold', pad=10)
#         ax2.set_ylabel('Sales Amount', fontsize=12)
        
#         ax2.text(0.01, 0.95, f"Total Predicted: {future_total:,.2f}", transform=ax2.transAxes, 
#                  fontsize=11, fontweight='bold', color='blue', va='top', bbox=box_props)
                 
#         ymin, ymax = ax2.get_ylim()
#         ax2.set_ylim(0, ymax * 1.35) # Increased to 1.35 for vertical text headroom
#         ax2.legend(loc='upper right')
#         ax2.grid(True, axis='y', alpha=0.3)

#     fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98) 
#     axes[-1].set_xlabel('Day of the Month', fontsize=12)
#     axes[-1].set_xticks(future_df['Day'])

#     plt.tight_layout(rect=[0, 0, 1, 0.96]) 
#     plt.show()



def plot_forecast(past_df, future_df, title, plot_type='line'):
    # COMMENTED OUT: We force has_past to False so it ignores the last year's data
    # has_past = not past_df.empty
    has_past = False 
    
    past_total = 0
    # COMMENTED OUT: Past actual data calculations
    # if has_past:
    #     past_df = past_df.copy()
    #     past_df['Day'] = past_df['Date'].dt.day
    #     past_total = past_df['Total Gross Amount'].sum()
        
    future_df = future_df.copy()
    future_df['Day'] = future_df['Date'].dt.day
    future_total = future_df['Predicted_Sales'].sum()

    # Fixed to 1 plot since we only want the predicted graph
    num_plots = 1 # 2 if has_past else 1
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(14, 6), sharex=True)
    
    # Ensure axes behaves as a list even with a single plot
    axes = [axes]

    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='lightgray')

    if plot_type == 'line':
        # --- PAST DATA (LINE) COMMENTED OUT ---
        # if has_past:
        #     ax1 = axes[0]
        #     ax1.plot(past_df['Day'], past_df['Total Gross Amount'], label='Actual (Last Year)', color='gray', marker='o', linestyle='dashed')
        #     for x, y in zip(past_df['Day'], past_df['Total Gross Amount']):
        #         ax1.text(x, y + (past_df['Total Gross Amount'].max() * 0.02), f"{y:.2f}", ha='center', va='bottom', fontsize=8, color='dimgray', rotation=90)
        #     ax1.set_title("Actual Data (Last Year)", fontsize=13, fontweight='bold', pad=10)
        #     ax1.set_ylabel('Sales Amount', fontsize=12)
        #     ax1.text(0.01, 0.95, f"Total Actual: {past_total:,.2f}", transform=ax1.transAxes, 
        #              fontsize=11, fontweight='bold', color='dimgray', va='top', bbox=box_props)
        #     ymin, ymax = ax1.get_ylim()
        #     ax1.set_ylim(ymin, ymax * 1.35) 
        #     ax1.legend(loc='upper right')
        #     ax1.grid(True, alpha=0.3)
        #     ax2 = axes[1]
        # else:
        
        ax2 = axes[0] # Directly assign the single plot area

        ax2.plot(future_df['Day'], future_df['Predicted_Sales'], label='Predicted (Future)', color='blue', marker='o', linewidth=2)
        for x, y in zip(future_df['Day'], future_df['Predicted_Sales']):
            # Added rotation=90
            ax2.text(x, y + (future_df['Predicted_Sales'].max() * 0.02), f"{y:.2f}", ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold', rotation=90)
            
        ax2.set_title("Predicted Data (Future)", fontsize=13, fontweight='bold', pad=10)
        ax2.set_ylabel('Sales Amount', fontsize=12)
        
        ax2.text(0.01, 0.95, f"Total Predicted: {future_total:,.2f}", transform=ax2.transAxes, 
                 fontsize=11, fontweight='bold', color='blue', va='top', bbox=box_props)
                 
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(ymin, ymax * 1.35) # Increased to 1.35 for vertical text headroom
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

    elif plot_type == 'bar':
        width = 0.5
        
        # --- PAST DATA (BAR) COMMENTED OUT ---
        # if has_past:
        #     ax1 = axes[0]
        #     bars_past = ax1.bar(past_df['Day'], past_df['Total Gross Amount'], width=width, label='Actual (Last Year)', color='lightgray')
        #     for bar in bars_past:
        #         yval = bar.get_height()
        #         ax1.text(bar.get_x() + bar.get_width()/2, yval + (past_df['Total Gross Amount'].max() * 0.02), f"{yval:.2f}", ha='center', va='bottom', fontsize=8, color='dimgray', rotation=90)
        #     ax1.set_title("Actual Data (Last Year)", fontsize=13, fontweight='bold', pad=10)
        #     ax1.set_ylabel('Sales Amount', fontsize=12)
        #     ax1.text(0.01, 0.95, f"Total Actual: {past_total:,.2f}", transform=ax1.transAxes, 
        #              fontsize=11, fontweight='bold', color='dimgray', va='top', bbox=box_props)
        #     ymin, ymax = ax1.get_ylim()
        #     ax1.set_ylim(0, ymax * 1.35) 
        #     ax1.legend(loc='upper right')
        #     ax1.grid(True, axis='y', alpha=0.3)
        #     ax2 = axes[1]
        # else:
        
        ax2 = axes[0] # Directly assign the single plot area

        bars_future = ax2.bar(future_df['Day'], future_df['Predicted_Sales'], width=width, label='Predicted (Future)', color='blue')
        for bar in bars_future:
            yval = bar.get_height()
            # Added rotation=90
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (future_df['Predicted_Sales'].max() * 0.02), f"{yval:.2f}", ha='center', va='bottom', fontsize=8, color='blue', fontweight='bold', rotation=90)
            
        ax2.set_title("Predicted Data (Future)", fontsize=13, fontweight='bold', pad=10)
        ax2.set_ylabel('Sales Amount', fontsize=12)
        
        ax2.text(0.01, 0.95, f"Total Predicted: {future_total:,.2f}", transform=ax2.transAxes, 
                 fontsize=11, fontweight='bold', color='blue', va='top', bbox=box_props)
                 
        ymin, ymax = ax2.get_ylim()
        ax2.set_ylim(0, ymax * 1.35) # Increased to 1.35 for vertical text headroom
        ax2.legend(loc='upper right')
        ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98) 
    axes[-1].set_xlabel('Day of the Month', fontsize=12)
    axes[-1].set_xticks(future_df['Day'])

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.show()
    
# ------ give the previous year dates for the selected month, using this in graph (*paused rn)  

def get_past_data(daily_sales_df, month, year):
    past_year = year - 1
    return daily_sales_df[(daily_sales_df['Month'] == month) & (daily_sales_df['Year'] == past_year)]

# ----------------------------------------------------------- total sales prediction in month

def predict_full_month(model, features, month, year, daily_sales_df, show_graph):
    prediction_df, start_date = generate_future_features(month, year)
    prediction_df['Predicted_Sales'] = model.predict(prediction_df[features]).round(2)
    
    print(f"\n================ FULL MONTH PREDICTIONS FOR {start_date[:7]} ================")
    print(prediction_df[['Date', 'Day_Name', 'Predicted_Sales']].to_string(index=False))
    print(f"\nTotal Predicted Monthly Sales: {prediction_df['Predicted_Sales'].sum():.2f}")
    print("========================================================================")
    
    if show_graph:
        past_data = get_past_data(daily_sales_df, month, year)
        plot_forecast(past_data, prediction_df, f"Monthly Trend: {start_date[:7]}", plot_type='line')

# ------------------------------------------------------------- weekend sales prediction

def predict_weekend_sales(model, features, month, year, daily_sales_df, show_graph):
    prediction_df, start_date = generate_future_features(month, year)
    prediction_df['Predicted_Sales'] = model.predict(prediction_df[features]).round(2)
    weekend_df = prediction_df[prediction_df['Is_Weekend'] == 1].reset_index(drop=True)
    
    print(f"\n>>>>>>>>>>>>>>> WEEKEND PREDICTIONS FOR {start_date[:7]} >>>>>>>>>>>>>>>")
    print(weekend_df[['Date', 'Day_Name', 'Predicted_Sales']].to_string(index=False))
    print(f"\nTotal Predicted Weekend Sales: {weekend_df['Predicted_Sales'].sum():.2f}")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    
    if show_graph:
        past_data = get_past_data(daily_sales_df, month, year)
        past_weekend = past_data[past_data['Is_Weekend'] == 1] if not past_data.empty else past_data
        plot_forecast(past_weekend, weekend_df, f"Weekend Prediction: {start_date[:7]}", plot_type='bar')

# ------------------------------------------------------------------------- weekday sales prediction

def predict_weekday_sales(model, features, month, year, daily_sales_df, show_graph):
    prediction_df, start_date = generate_future_features(month, year)
    prediction_df['Predicted_Sales'] = model.predict(prediction_df[features]).round(2)
    weekday_df = prediction_df[prediction_df['Is_Weekend'] == 0].reset_index(drop=True)
    
    print(f"\n--------------- WEEKDAY PREDICTIONS FOR {start_date[:7]} ---------------")
    print(weekday_df[['Date', 'Day_Name', 'Predicted_Sales']].to_string(index=False))
    print(f"\nTotal Predicted Weekday Sales: {weekday_df['Predicted_Sales'].sum():.2f}")
    print("------------------------------------------------------------------")
    
    if show_graph:
        past_data = get_past_data(daily_sales_df, month, year)
        past_weekday = past_data[past_data['Is_Weekend'] == 0] if not past_data.empty else past_data
        plot_forecast(past_weekday, weekday_df, f"Weekday Comparison: {start_date[:7]}", plot_type='line')

# ---------------------------------------------------- custom date prediction

def predict_custom_dates(model, features, month, year, dates_input, daily_sales_df, show_graph):
    try:
        requested_days = [int(day.strip()) for day in dates_input.split(',')]
    except ValueError:
        print("Error: Invalid date format. Please use numbers separated by commas.")
        return

    prediction_df, start_date = generate_future_features(month, year)
    custom_df = prediction_df[prediction_df['Date'].dt.day.isin(requested_days)].reset_index(drop=True)
    
    if custom_df.empty:
        print("Error: None of the entered dates are valid for this month.")
        return

    custom_df['Predicted_Sales'] = model.predict(custom_df[features]).round(2)
    
    print(f"\n*************** CUSTOM DATES PREDICTION FOR {start_date[:7]} ***************")
    print(custom_df[['Date', 'Day_Name', 'Predicted_Sales']].to_string(index=False))
    print(f"\nTotal Predicted Sales for Selected Dates: {custom_df['Predicted_Sales'].sum():.2f}")
    print("******************************************************************")
    
    if show_graph:
        past_data = get_past_data(daily_sales_df, month, year)
        past_custom = past_data[past_data['Date'].dt.day.isin(requested_days)] if not past_data.empty else past_data
        plot_forecast(past_custom, custom_df, f"Custom Dates Comparison: {start_date[:7]}", plot_type='bar')


# --------------------------------------------------- 4 v 5 weekend

def simulate_weekend_impact(model, features, month, year, daily_sales_df, show_graph):
    historical_data = daily_sales_df[(daily_sales_df['Month'] == month) & (daily_sales_df['Year'] == year)]
    
    if not historical_data.empty:
        print(f"\n Previous data found. Using ACTUAL sales data for {month}/{year} simulation...")
        df = historical_data.copy()
        df['Sales_Data'] = df['Total Gross Amount'] 
    else:
        print(f"\n Future date detected. Using PREDICTIVE model for {month}/{year} simulation...")
        df, _ = generate_future_features(month, year)
        df['Sales_Data'] = model.predict(df[features])

    # 1. Calculate the real calendar baseline
    weekend_days = df[df['Is_Weekend'] == 1]
    weekday_days = df[df['Is_Weekend'] == 0]
    
    actual_weekend_count = len(weekend_days) // 2 
    total_sales = df['Sales_Data'].sum()
    
    avg_weekend_day = weekend_days['Sales_Data'].mean()
    avg_weekday_day = weekday_days['Sales_Data'].mean()
    weekend_premium = (avg_weekend_day - avg_weekday_day) * 2

    if actual_weekend_count == 4:
        sales_4w = total_sales
        sales_5w = total_sales + weekend_premium
    elif actual_weekend_count == 5:
        sales_5w = total_sales
        sales_4w = total_sales - weekend_premium
    else:
        sales_4w = total_sales
        sales_5w = total_sales + weekend_premium 

    # 4. Terminal Output
    print(f"\n====== BUSINESS IMPACT SIMULATOR: {month}/{year} ======")
    print(f"Calendar Reality: This month naturally has {actual_weekend_count} weekends.")
    print(f"Average Weekday Sales: {avg_weekday_day:.2f} / day")
    print(f"Average Weekend Sales: {avg_weekend_day:.2f} / day")
    print(f"Financial Value of 1 extra weekend: +{weekend_premium:.2f}")
    print("-" * 55)
    
    if actual_weekend_count == 4:
        print(f"---> Simulated Total with 4 Weekends: {sales_4w:.2f}  (Actual Calendar)")
        print(f"     Simulated Total with 5 Weekends: {sales_5w:.2f}")
    else:
        print(f"     Simulated Total with 4 Weekends: {sales_4w:.2f}")
        print(f"---> Simulated Total with 5 Weekends: {sales_5w:.2f}  (Actual Calendar)")
    print("=======================================================\n")

    # 5. Graphing Logic
    if show_graph:
        plt.figure(figsize=(10, 6))
        
        labels = ['4 Weekends', '5 Weekends']
        values = [sales_4w, sales_5w]
        
        colors = ['blue' if actual_weekend_count == 4 else 'lightgray',
                  'blue' if actual_weekend_count == 5 else 'lightgray']
        
        bars = plt.bar(labels, values, color=colors, width=0.5)
       
        for bar in bars:
            yval = bar.get_height()
        
            plt.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", 
                     ha='center', va='bottom', fontsize=11, fontweight='bold', color='dimgray')
            
        plt.title(f"4 vs 5 Weekend: {month}/{year}", fontsize=14, fontweight='bold')
        plt.ylabel('Total Monthly Sales', fontsize=12)
        plt.text(0.5, max(values) * 1.05, f"for 5th Weekend:\n+{weekend_premium:.2f}", 
                 horizontalalignment='center', fontsize=12, bbox=dict(facecolor='yellow', alpha=0.3))
        
        ymin, ymax = plt.ylim()
    
        plt.ylim(ymin, ymax * 1.20)
        
        plt.grid(True, axis='y', alpha=0.3)
        plt.show() 

# ---------------------------------------------------------- prediction menu        

def prediction_menu(daily_sales_df):
    model, features = train_model(daily_sales_df)
    
    while True:
        print("\nOptions: 'monthly', 'weekend', 'weekday', 'custom', 'impact', or 'exit'")
        choice = input("What would you like to do? ").strip().lower()
        
        if choice == 'exit':
            print("Exiting prediction module.")
            break
            
        if choice not in ['monthly', 'weekend', 'weekday', 'custom', 'impact']:
            print("Invalid choice. Please type one of the exact options.")
            continue
            
        month_input = input("Enter Month (1-12): ").strip()
        year_input = input("Enter Year (e.g., 2026, 2027): ").strip()
        
        # We ask for the graph for EVERY option, including 'impact'
        graph_input = input("Show graph? (y/n): ").strip().lower()
        show_graph = graph_input == 'y'
        
        try:
            month = int(month_input)
            year = int(year_input)
            
            if not (1 <= month <= 12):
                print("Error: Month must be between 1 and 12.")
                continue
                
            if choice == 'monthly':
                predict_full_month(model, features, month, year, daily_sales_df, show_graph)
            elif choice == 'weekend':
                predict_weekend_sales(model, features, month, year, daily_sales_df, show_graph)
            elif choice == 'weekday':
                predict_weekday_sales(model, features, month, year, daily_sales_df, show_graph)
            elif choice == 'custom':
                dates_input = input("Enter dates separated by comma (e.g., 1, 2, 6, 8): ").strip()
                predict_custom_dates(model, features, month, year, dates_input, daily_sales_df, show_graph)
            elif choice == 'impact':
                simulate_weekend_impact(model, features, month, year, daily_sales_df, show_graph)
                
        except ValueError:
            print("Invalid input. Please type numbers only for month and year.")
            
# ---------------------------------------- new files

user_input = input("new filess? (y/n): ").strip().lower() # combines all the files

if user_input == 'y':
    merged_df, daily_sales = merge_and_process_data()
else:
    print("Continuing further")

# ------------------------------------------- for prediction

if __name__ == "__main__":
    print("Loading previous merged file...")
    try:
        daily_sales = pd.read_csv(r"E:\wovVRA Sales forecasting model\daily_sales_clean.csv", parse_dates=['Date'])
        print("Files loaded successfully.")
        
        # Trigger the interactive menu
        prediction_menu(daily_sales)
        
    except FileNotFoundError:
        print("Error: clean CSV not found. Run the data processing script first.")