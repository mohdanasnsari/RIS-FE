import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

def combine_master_data():   #This combines 366 days data into single file
    print("\n" + "="*50)
    print(" 📂 DATA COMBINER: 366-DAY MASTER FILE")
    print("="*50)

    # 1. Get the folder location
    folder_path = input(r"Enter the folder path containing all your Excel sheets: ").strip()
    search_pattern = os.path.join(folder_path, "*.xlsx")
    
    # Grab all excel files, ignoring hidden temporary files
    files = [f for f in glob.glob(search_pattern) if not os.path.basename(f).startswith("~$")]
    
    if not files:
        print("[!] No Excel files found in that directory. Please check the path.")
        return

    print(f"\nFound {len(files)} files. Stitching them together... This might take a moment depending on the data size.\n")

    # 2. Read and combine all files
    df_list = []
    for file in files:
        print(f"Reading: {os.path.basename(file)}")
        try:
            df = pd.read_excel(file)
            df_list.append(df)
        except Exception as e:
            print(f"  [!] Could not read {os.path.basename(file)}: {e}")

    if not df_list:
        print("[!] No data could be extracted.")
        return

    master_df = pd.concat(df_list, ignore_index=True)
    
    # 3. Basic cleanup (strip accidental spaces from column headers)
    master_df.columns = master_df.columns.str.strip()

    # 4. Save the combined file
    print("\n" + "="*50)
    save_folder = input(r"Enter the exact folder path where you want to save the final Master CSV: ").strip()
    save_name = input("Enter the file name (e.g., Master_366_Days.csv): ").strip()
    
    if not save_name.lower().endswith('.csv'):
        save_name += '.csv'
        
    save_path = os.path.join(save_folder, save_name)
    
    master_df.to_csv(save_path, index=False)
    print(f"\n[+] BOOM! Success. Master file with {len(master_df):,} rows saved to: {save_path}")

def calculate_monthly_sales_with_mall():    #calculated monthly sales of each brand in mall
    print("\n" + "="*50)
    print(" 📊 MONTHLY SALES CALCULATOR (WITH MALL & BRANDS)")
    print("="*50)

    # 1. Load the Master File
    file_path = input(r"Enter the exact path to your Master_366_Days.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found. Please check the path.")
        return

    # 2. Strip any hidden spaces from column headers
    df.columns = df.columns.str.strip()
    
    # 3. Handle variations in column names
    mall_col = 'Mall' if 'Mall' in df.columns else 'Mall Name'
    brand_col = 'Brand Name' if 'Brand Name' in df.columns else 'Tenant Name'
    sales_col = 'Total Gross Amount' if 'Total Gross Amount' in df.columns else 'Gross Amount'
    
    # Check for required columns
    required_cols = ['Date', brand_col, sales_col]
    
    if mall_col in df.columns:
        required_cols.append(mall_col)
        group_index = [mall_col, brand_col]
    else:
        print(f"[!] Warning: Could not find a 'Mall' column. Grouping by Brand only.")
        group_index = [brand_col]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[!] Error: Missing required columns: {missing_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    print("\nProcessing 366 days of data... Please wait.")

    # 4. Clean Dates
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    
    # 5. CRITICAL FIX: Force the sales column to be numeric, removing any commas
    df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(',', ''), errors='coerce')
    
    # 6. Drop rows with missing critical data
    df = df.dropna(subset=required_cols) 
    
    # 7. Extract Month
    df['Month'] = df['Date'].dt.to_period('M')

    # 8. Generate Pivot Table
    monthly_sales = df.pivot_table(
        index=group_index,
        columns='Month',
        values=sales_col,
        aggfunc='sum',
        fill_value=0
    )

    # Clean up column headers (e.g., 'May 2025')
    monthly_sales.columns = [col.strftime('%b %Y') for col in monthly_sales.columns]
    monthly_sales = monthly_sales.reset_index()

    # 9. Save the final report
    print("\n" + "="*50)
    save_folder = input(r"Enter the folder path to save the Monthly Sales report: ").strip()
    save_path = os.path.join(save_folder, "Master_Monthly_Sales_Report.csv")
    
    monthly_sales.to_csv(save_path, index=False)
    print(f"\n[+] Success! Monthly sales report saved to: {save_path}")

def filter_csv_by_mall(file_path=None, target_mall=None):  #Helps in filtering mall
    print("\n" + "="*50)
    print(" 🏢 UNIVERSAL MALL FILTER UTILITY")
    print("="*50)

    if not file_path:
        file_path = input(r"Enter the exact path to your CSV file: ").strip()
    
    if not target_mall:
        target_mall = input("Enter the Mall Name to filter by (e.g., Fiza by Nexus): ").strip()

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found. Please check the path.")
        return None

    # Strip hidden spaces from headers
    df.columns = df.columns.str.strip()
    
    # Identify the correct Mall column
    mall_col = None
    for col in ['Mall', 'Mall Name', 'Mall_Name']:
        if col in df.columns:
            mall_col = col
            break
            
    if not mall_col:
        print(f"[!] Error: Could not find a 'Mall' column. Found columns: {df.columns.tolist()}")
        return None

    # Apply the filter (case-insensitive)
    filtered_df = df[df[mall_col].astype(str).str.contains(target_mall, case=False, na=False)]
    
    if filtered_df.empty:
        print(f"[!] No records found for mall: '{target_mall}'.")
        return None

    # Save the filtered file
    save_folder = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace(".csv", "")
    safe_mall_name = target_mall.replace(" ", "_").replace("/", "-")
    
    save_path = os.path.join(save_folder, f"{base_name}_Filtered_{safe_mall_name}.csv")
    
    filtered_df.to_csv(save_path, index=False)
    print(f"\n[+] Success! Filtered file ({len(filtered_df)} rows) saved to:\n    {save_path}")
    
    return save_path

def combine_invoice_data():   #Function to combine invoives that I downloaded
    folder_path = input(r"Enter folder path containing your invoice Excel sheets: ").strip()
    search_pattern = os.path.join(folder_path, "*.xlsx")
    files = [f for f in glob.glob(search_pattern) if not os.path.basename(f).startswith("~$")]
    
    if not files:
        print("[!] No Excel files found.")
        return

    print(f"Combining {len(files)} invoice files...")
    
    df_list = []
    for file in files:
        try:
            df = pd.read_excel(file)
            df_list.append(df)
        except Exception as e:
            print(f"  [!] Error reading {os.path.basename(file)}: {e}")

    if not df_list:
        return

    master_df = pd.concat(df_list, ignore_index=True)
    master_df.columns = master_df.columns.str.strip()

    save_folder = input(r"Enter the folder path to save the Combined Invoices CSV: ").strip()
    save_name = input("Enter the file name (e.g., Master_Invoices.csv): ").strip()
    
    if not save_name.lower().endswith('.csv'):
        save_name += '.csv'
        
    save_path = os.path.join(save_folder, save_name)
    
    master_df.to_csv(save_path, index=False)
    print(f"\n[+] Success! Combined invoices ({len(master_df)} rows) saved to: {save_path}")


def calculate_daily_sales_invoice():   #calculated total daily sale of brand and total invoices
    print("\n" + "="*50)
    print(" 📅 DAILY SALES & INVOICE CALCULATOR")
    print("="*50)

    # USER INPUT 1: Source File
    file_path = input(r"Enter the exact path to your Master_Invoices.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found. Please check the path.")
        return

    df.columns = df.columns.str.strip()
    
    # Handle column variations
    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    sales_col = 'Gross Amount' if 'Gross Amount' in df.columns else 'Total Gross Amount'
    inv_col = 'Invoice Number' if 'Invoice Number' in df.columns else 'Invoice Nu'
    
    if 'Date' not in df.columns or brand_col not in df.columns or sales_col not in df.columns or inv_col not in df.columns:
        print(f"[!] Error: Missing required columns (Need Date, Brand, Sales, and Invoice Number).")
        print(f"Found columns: {df.columns.tolist()}")
        return

    print("Processing daily data... Please wait.")

    # Clean data
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Date', brand_col, sales_col]) 

    # Aggregate both Sales (sum) and Invoices (nunique)
    daily_sales = df.groupby([brand_col, 'Date']).agg(
        Total_Gross_Amount=(sales_col, 'sum'),
        Total_Invoices=(inv_col, 'nunique')
    ).reset_index()
    
    daily_sales['Date'] = daily_sales['Date'].dt.strftime('%d-%b-%Y')

    print("\n" + "-"*50)
    # USER INPUT 2: Save Location
    save_folder = input(r"Enter the exact folder path to save the Daily report: ").strip()
    save_name = input("Enter file name (e.g., Daily_Sales_Report.csv): ").strip()
    
    if not save_name.lower().endswith('.csv'):
        save_name += '.csv'
        
    save_path = os.path.join(save_folder, save_name)
    daily_sales.to_csv(save_path, index=False)
    
    print(f"\n[+] Success! Daily report saved to:\n    {save_path}")



def calculate_monthly_sales_invoice():   #Calculated monthly total sale of brand and total invoices in month
    print("\n" + "="*50)
    print(" 📊 MONTHLY SALES & INVOICE CALCULATOR")
    print("="*50)

    # USER INPUT 1: Source File
    file_path = input(r"Enter the exact path to your Master_Invoices.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found. Please check the path.")
        return

    df.columns = df.columns.str.strip()
    
    # Handle column variations
    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    sales_col = 'Gross Amount' if 'Gross Amount' in df.columns else 'Total Gross Amount'
    inv_col = 'Invoice Number' if 'Invoice Number' in df.columns else 'Invoice Nu'
    
    if 'Date' not in df.columns or brand_col not in df.columns or sales_col not in df.columns or inv_col not in df.columns:
        print(f"[!] Error: Missing required columns (Need Date, Brand, Sales, and Invoice Number).")
        print(f"Found columns: {df.columns.tolist()}")
        return

    print("Processing monthly data... Please wait.")

    # Clean data
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    df[sales_col] = pd.to_numeric(df[sales_col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['Date', brand_col, sales_col]) 
    
    df['Month'] = df['Date'].dt.to_period('M')

    # Aggregate both Sales (sum) and Invoices (nunique)
    monthly_sales = df.groupby([brand_col, 'Month']).agg(
        Total_Gross_Amount=(sales_col, 'sum'),
        Total_Invoices=(inv_col, 'nunique')
    ).reset_index()

    # Format the month back to a readable string (e.g., 'May 2025')
    monthly_sales['Month'] = monthly_sales['Month'].dt.strftime('%b %Y')

    print("\n" + "-"*50)
    # USER INPUT 2: Save Location
    save_folder = input(r"Enter the exact folder path to save the Monthly report: ").strip()
    save_name = input("Enter file name (e.g., Monthly_Sales_Summary.csv): ").strip()
    
    if not save_name.lower().endswith('.csv'):
        save_name += '.csv'
        
    save_path = os.path.join(save_folder, save_name)
    monthly_sales.to_csv(save_path, index=False)
    
    print(f"\n[+] Success! Monthly report saved to:\n    {save_path}")


# Simulation starts here


## Case 1: if 10% price increased
def run_price_simulator():   
    print("\n" + "="*50)
    print(" 📈 SCENARIO 1: PRICE ELASTICITY SIMULATOR")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    
    if brand_col not in df.columns or 'Total_Gross_Amount' not in df.columns or 'Total_Invoices' not in df.columns:
        print("[!] Error: Missing required columns in the daily report.")
        return

    print("\nCalculating 65-day baselines... Please wait.")

    # 1. Calculate the exact ATV for each individual day
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)

    # 2. Find the median ATV and Volume across the April 5 - June 8 period
    baselines = df.groupby(brand_col).agg(
        Median_ATV=('Daily_ATV', 'median'),
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    baselines['Baseline_Daily_Turnover'] = baselines['Median_ATV'] * baselines['Median_Volume']

    # 3. Display Selection Menu
    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter the number of the brand to simulate (or 'exit'): ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        
        print(f"\n--- Current Baseline for {selected_brand} ---")
        print(f"Median Transaction Value (ATV): ₹ {brand_data['Median_ATV']:,.2f}")
        print(f"Median Daily Invoices:          {brand_data['Median_Volume']:,.0f}")
        print(f"Baseline Daily Turnover:        ₹ {brand_data['Baseline_Daily_Turnover']:,.2f}")
        
        # 4. Scenario Inputs
        price_change = float(input("\nEnter Price Increase % (e.g., 10 for 10% hike): "))
        volume_drop = float(input("Enter Expected Customer Drop % (e.g., 5 for 5% loss): "))
        
        # 5. Math & Output
        new_atv = brand_data['Median_ATV'] * (1 + (price_change / 100))
        new_volume = brand_data['Median_Volume'] * (1 - (volume_drop / 100))
        new_turnover = new_atv * new_volume
        diff = new_turnover - brand_data['Baseline_Daily_Turnover']
        
        print("\n" + "-"*45)
        print(f" SIMULATION RESULTS: {selected_brand}")
        print("-" * 45)
        print(f"Projected ATV:           ₹ {new_atv:,.2f}")
        print(f"Projected Daily Volume:  {new_volume:,.1f} invoices")
        print(f"Projected Turnover:      ₹ {new_turnover:,.2f} / day")
        
        if diff >= 0:
            print(f"\n✅ Outcome: POSITIVE (Gain of ₹ {diff:,.2f} per day)")
        else:
            print(f"\n❌ Outcome: NEGATIVE (Loss of ₹ {abs(diff):,.2f} per day)")
        print("-" * 45)

        # 6. Optional Graph
        if input("\nShow visual comparison graph? (y/n): ").strip().lower() == 'y':
            plot_impact_graph(selected_brand, brand_data['Baseline_Daily_Turnover'], new_turnover, diff)

    except (ValueError, IndexError):
        print("\n[!] Invalid input. Simulation aborted.")

def plot_impact_graph(brand, baseline, new_val, diff):   #10% price hike graph
    plt.figure(figsize=(8, 6))
    labels = ['Current Baseline', 'Projected Future']
    values = [baseline, new_val]
    
    sim_color = 'green' if diff >= 0 else 'red'
    bars = plt.bar(labels, values, color=['lightgray', sim_color], width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max(values) * 0.01), 
                 f"₹ {yval:,.2f}", ha='center', va='bottom', fontsize=11, fontweight='bold', color='dimgray')
    
    plt.title(f"What if a specific brand increases its prices by 10%: {brand}", fontsize=14, fontweight='bold')
    plt.ylabel("Daily Turnover (₹)")
    
    sign = "+" if diff >= 0 else ""
    plt.text(0.5, max(values) * 1.05, f"Projected Impact:\n{sign}₹ {diff:,.2f}", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.25) 
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()



## Case 2: If customer shift online
def run_online_shift_simulator():
    print("\n" + "="*50)
    print(" 💻 SCENARIO 2: ONLINE SHOPPING SHIFT SIMULATOR")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    
    if brand_col not in df.columns or 'Total_Gross_Amount' not in df.columns or 'Total_Invoices' not in df.columns:
        print("[!] Error: Missing required columns.")
        return

    # Calculate Baselines
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)
    
    baselines = df.groupby(brand_col).agg(
        Median_ATV=('Daily_ATV', 'median'),
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number: ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        baseline_vol = brand_data['Median_Volume']
        baseline_atv = brand_data['Median_ATV']
        
        baseline_monthly_turnover = baseline_vol * baseline_atv * 30 
        baseline_monthly_transactions = baseline_vol * 30
        
        print(f"\n--- Current Baseline for {selected_brand} ---")
        print(f"Monthly Transactions: {baseline_monthly_transactions:,.0f} invoices")
        print(f"Monthly Turnover:     ₹ {baseline_monthly_turnover:,.2f}")
        
        # Inputs
        monthly_drop = float(input("\nEnter Expected Monthly Customer Drop % (e.g., 2 for 2%): "))
        months_to_simulate = int(input("How many months into the future? (e.g., 12): "))
        
        # Compounding Math
        projection_data = []
        current_vol = baseline_vol
        
        for month in range(1, months_to_simulate + 1):
            current_vol = current_vol * (1 - (monthly_drop / 100))
            current_turnover = current_vol * baseline_atv * 30
            current_transactions = current_vol * 30
            
            projection_data.append({
                'Month': f"M{month}",
                'Monthly_Transactions': current_transactions,
                'Monthly_Turnover': current_turnover,
            })
            
            if month in [1, months_to_simulate]: 
                print(f"Month {month:02d}: {current_transactions:,.0f} transactions | ₹ {current_turnover:,.2f}")

        # Graph
        if input("\nShow trend graph? (y/n): ").strip().lower() == 'y':
            plot_online_shift(selected_brand, baseline_monthly_turnover, baseline_monthly_transactions, projection_data)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_online_shift(brand, base_turnover, base_transactions, data):  # online shift graph
    months = [d['Month'] for d in data]
    turnovers = [d['Monthly_Turnover'] for d in data]
    transactions = [d['Monthly_Transactions'] for d in data]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Axis: Turnover (Red)
    color1 = 'tab:red'
    ax1.set_xlabel('Months')
    ax1.set_ylabel('Monthly Turnover (₹)', color=color1, fontweight='bold')
    ax1.plot(months, turnovers, color=color1, marker='o', label='Projected Turnover', linewidth=2)
    ax1.plot(months, [base_turnover]*len(months), color=color1, linestyle='--', alpha=0.3)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Right Axis: Transactions (Blue)
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Total Monthly Transactions', color=color2, fontweight='bold')
    ax2.plot(months, transactions, color=color2, marker='s', label='Projected Transactions', linewidth=2)
    ax2.plot(months, [base_transactions]*len(months), color=color2, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title(f"Online Shift Impact: {brand}\nTurnover vs. Transaction Count", fontsize=14, fontweight='bold')
    
    # Text Box Summary
    final_trans_loss = base_transactions - transactions[-1]
    plt.text(0.05, 0.05, f"Final Month Deficit:\n-{final_trans_loss:,.0f} Transactions", 
             transform=ax1.transAxes, fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
             
    fig.tight_layout()
    plt.grid(True, alpha=0.2)
    plt.show()


## Case 6: If customer trend continues to grow next quarter
def run_trend_projection_simulator():
    print("\n" + "="*50)
    print(" 🚀 SCENARIO 3: NEXT QUARTER TREND PROJECTION")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    
    if brand_col not in df.columns or 'Total_Gross_Amount' not in df.columns:
        print("[!] Error: Missing required columns.")
        return

    # 1. Calculate a "Normalized" 30-Day Baseline
    # We use the median daily sales to ignore extremely high/low days, then multiply by 30
    baselines = df.groupby(brand_col).agg(
        Median_Daily_Sales=('Total_Gross_Amount', 'median')
    ).reset_index()
    
    baselines['Normalized_Monthly_Baseline'] = baselines['Median_Daily_Sales'] * 30

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to simulate (or 'exit'): ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        baseline_month = brand_data['Normalized_Monthly_Baseline']
        
        print(f"\n--- Baseline for {selected_brand} ---")
        print(f"Normalized Monthly Turnover: ₹ {baseline_month:,.2f}")
        
        # 2. Scenario Inputs
        growth_trend = float(input("\nEnter Projected Monthly Growth % (e.g., 5 for +5% MoM): "))
        
        # 3. Project the Next Quarter (3 Months)
        print("\n" + "-"*45)
        print(f" 📈 Q3 PROJECTION REPORT ({growth_trend}% MoM Growth)")
        print("-" * 45)
        
        projection_data = []
        current_turnover = baseline_month
        total_quarter_sales = 0
        
        for month in range(1, 4):
            # Compound the growth
            current_turnover = current_turnover * (1 + (growth_trend / 100))
            total_quarter_sales += current_turnover
            
            projection_data.append({
                'Month': f"Projected Month {month}",
                'Turnover': current_turnover
            })
            print(f"Projected Month {month}: ₹ {current_turnover:,.2f}")

        # Baseline Quarter (if there was 0% growth) for comparison
        baseline_quarter = baseline_month * 3
        net_gain = total_quarter_sales - baseline_quarter

        print("-" * 45)
        print(f"Total Projected Quarter: ₹ {total_quarter_sales:,.2f}")
        print(f"Extra Revenue vs Flat:   +₹ {net_gain:,.2f}")
        print("-" * 45)

        # 4. Optional Graph
        if input("\nShow projection graph? (y/n): ").strip().lower() == 'y':
            plot_trend_projection(selected_brand, baseline_month, projection_data)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_trend_projection(brand, baseline, data):  # customer trend grows graph 90 days
    months = ['Current Baseline'] + [d['Month'] for d in data]
    turnovers = [baseline] + [d['Turnover'] for d in data]
    
    plt.figure(figsize=(9, 6))
    
    # Plot bars
    bars = plt.bar(months, turnovers, color=['lightgray', '#4CAF50', '#4CAF50', '#4CAF50'], width=0.6)
    
    # Add a flat line showing what would happen with 0% growth
    plt.plot([0, 3], [baseline, baseline], color='gray', linestyle='--', linewidth=2, label='Flat Baseline (0% Growth)')
    
    # Data labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.02, 
                 f"₹ {yval/100000:,.1f}L", ha='center', va='bottom', fontsize=10, fontweight='bold', color='dimgray')
    
    plt.title(f"Next Quarter Growth Projection: {brand}", fontsize=14, fontweight='bold')
    plt.ylabel("Monthly Turnover (₹)", fontsize=12)
    plt.legend()
    
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.15) 
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


## Case 6: if customer spending habit changed
def run_macro_spending_simulator():
    print("\n" + "="*50)
    print(" 🌍 SCENARIO 4: MACRO SPENDING HABITS SIMULATOR")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    
    if brand_col not in df.columns or 'Total_Gross_Amount' not in df.columns or 'Total_Invoices' not in df.columns:
        print("[!] Error: Missing required columns.")
        return

    # 1. Calculate Baselines
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)
    
    baselines = df.groupby(brand_col).agg(
        Median_ATV=('Daily_ATV', 'median'),
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to simulate (or 'exit'): ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        
        base_atv = brand_data['Median_ATV']
        base_vol = brand_data['Median_Volume']
        base_daily_turnover = base_atv * base_vol
        base_monthly_turnover = base_daily_turnover * 30
        
        print(f"\n--- Current Baseline for {selected_brand} ---")
        print(f"Median ATV:              ₹ {base_atv:,.2f}")
        print(f"Median Daily Invoices:   {base_vol:,.0f}")
        print(f"Baseline Monthly Turnover: ₹ {base_monthly_turnover:,.2f}")
        
        # 2. Resilience Check & Macro Variables
        print("\n" + "-"*40)
        is_resilient = input("Is this brand 'recession-proof' (e.g., Luxury or Essentials)? (y/n): ").strip().lower()
        
        if is_resilient == 'y':
            print("\n🛡️ Resilient Brand Mode Active (Ignoring Macro Drops)")
            atv_shift = float(input("Enter normal expected ATV growth % (e.g., 10 for +10%): "))
            vol_shift = float(input("Enter normal expected Volume growth % (e.g., 2 for +2%): "))
            color_theme = 'tab:green'
        else:
            print("\n📉 Vulnerable Brand Mode Active (Assume economic downturn)")
            atv_shift = float(input("Enter expected % drop in Average Transaction Value (e.g., -10 for 10% drop): "))
            vol_shift = float(input("Enter expected % drop in Shopping Frequency (e.g., -5 for 5% drop): "))
            color_theme = 'tab:red'
        
        # 3. Math
        new_atv = base_atv * (1 + (atv_shift / 100))
        new_vol = base_vol * (1 + (vol_shift / 100))
        
        new_daily_turnover = new_atv * new_vol
        new_monthly_turnover = new_daily_turnover * 30
        monthly_diff = new_monthly_turnover - base_monthly_turnover
        
        # 4. Output
        print("\n" + "-"*45)
        print(f" 📊 IMPACT REPORT: {selected_brand}")
        print("-" * 45)
        print(f"Projected ATV:           ₹ {new_atv:,.2f}")
        print(f"Projected Daily Volume:  {new_vol:,.1f}")
        
        print(f"\nFinancial Impact:")
        if monthly_diff >= 0:
            print(f"✅ Monthly Surplus: +₹ {monthly_diff:,.2f}")
        else:
            print(f"❌ Monthly Deficit: -₹ {abs(monthly_diff):,.2f}")
        print("-" * 45)

        # 5. Graph
        if input("\nShow impact graph? (y/n): ").strip().lower() == 'y':
            plot_macro_impact(selected_brand, base_monthly_turnover, new_monthly_turnover, monthly_diff, color_theme)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_macro_impact(brand, baseline, projected, diff, color_theme): # customer consuming habit change graph
    plt.figure(figsize=(8, 6))
    
    labels = ['Baseline Monthly Sales', 'Projected Monthly Sales']
    values = [baseline, projected]
    
    bars = plt.bar(labels, values, color=['lightgray', color_theme], width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.02, 
                 f"₹ {yval/100000:,.2f}L", ha='center', va='bottom', fontweight='bold', color='dimgray')
    
    plt.title(f"Consumer spending habit: {brand}", fontsize=14, fontweight='bold')
    plt.ylabel("Total Monthly Turnover (₹)")
    
    sign = "+" if diff >= 0 else "-"
    plt.text(0.5, max(values) * 1.1, f"Net Monthly Impact:\n{sign}₹ {abs(diff):,.2f}", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor=color_theme))
    
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.25)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()



# Case 7:  Supply chain disrupts 
def run_supply_chain_simulator():
    print("\n" + "="*50)
    print(" ⚠️ SCENARIO: SUPPLY CHAIN DISRUPTION")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)
    
    baselines = df.groupby(brand_col).agg(
        Median_ATV=('Daily_ATV', 'median'),
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to simulate: ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        
        base_atv = brand_data['Median_ATV']
        base_vol = brand_data['Median_Volume']
        base_daily_turnover = base_atv * base_vol
        
        # Inputs
        duration = int(input("\nEnter duration of the supply chain crisis (in days): "))
        vol_drop = float(input("Enter % drop in Daily Volume (due to out-of-stock walkouts): "))
        atv_drop = float(input("Enter % drop in ATV (customers buying fewer items): "))
        
        # Math
        normal_period_revenue = base_daily_turnover * duration
        
        crisis_vol = base_vol * (1 - (vol_drop / 100))
        crisis_atv = base_atv * (1 - (atv_drop / 100))
        crisis_daily_turnover = crisis_vol * crisis_atv
        crisis_period_revenue = crisis_daily_turnover * duration
        
        lost_revenue = normal_period_revenue - crisis_period_revenue
        
        # Output
        print("\n" + "-"*45)
        print(f" 📉 SUPPLY CHAIN IMPACT: {selected_brand} ({duration} Days)")
        print("-" * 45)
        print(f"Normal Expected Revenue: ₹ {normal_period_revenue:,.2f}")
        print(f"Actual Crisis Revenue:   ₹ {crisis_period_revenue:,.2f}")
        print(f"Total Revenue Lost:      -₹ {lost_revenue:,.2f}")
        print("-" * 45)

        if input("\nShow impact graph? (y/n): ").strip().lower() == 'y':
            plot_short_term_impact(selected_brand, "Supply Chain Crisis", normal_period_revenue, crisis_period_revenue, lost_revenue)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_short_term_impact(brand, event_name, normal_rev, actual_rev, lost_rev):
    plt.figure(figsize=(8, 6))
    labels = ['Normal Expected', 'Actual Captured']
    values = [normal_rev, actual_rev]
    
    bars = plt.bar(labels, values, color=['#4CAF50', '#F44336'], width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.02, 
                 f"₹ {yval/1000:,.0f}K", ha='center', va='bottom', fontweight='bold', color='dimgray')
    
    plt.title(f"{event_name} Impact: {brand}", fontsize=14, fontweight='bold')
    plt.ylabel("Period Revenue (₹)")
    
    plt.text(0.5, max(values) * 1.1, f"Total Deficit:\n-₹ {lost_rev:,.2f}", 
             ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='red'))
    
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.25)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()



# Case 7: Competitor lower their price
def run_competitor_action_simulator():
    print("\n" + "="*50)
    print(" 🏷️ SCENARIO: COMPETITOR PRICE CUT")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)
    
    baselines = df.groupby(brand_col).agg(
        Median_ATV=('Daily_ATV', 'median'),
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to simulate: ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        
        base_atv = brand_data['Median_ATV']
        base_vol = brand_data['Median_Volume']
        base_daily_turnover = base_atv * base_vol
        
        # Inputs
        duration = int(input("\nEnter duration of competitor's sale (in days): "))
        attrition_rate = float(input("Enter % of daily invoices lost to the competitor: "))
        
        # Math (ATV stays normal, only volume drops)
        normal_period_revenue = base_daily_turnover * duration
        
        bleed_vol = base_vol * (1 - (attrition_rate / 100))
        bleed_daily_turnover = bleed_vol * base_atv
        actual_period_revenue = bleed_daily_turnover * duration
        
        lost_revenue = normal_period_revenue - actual_period_revenue
        lost_transactions = (base_vol - bleed_vol) * duration
        
        # Output
        print("\n" + "-"*45)
        print(f" 💸 COMPETITOR IMPACT: {selected_brand} ({duration} Days)")
        print("-" * 45)
        print(f"Total Transactions Lost: {lost_transactions:,.0f}")
        print(f"Normal Expected Revenue: ₹ {normal_period_revenue:,.2f}")
        print(f"Actual Captured Revenue: ₹ {actual_period_revenue:,.2f}")
        print(f"Revenue Bled to Rival:   -₹ {lost_revenue:,.2f}")
        print("-" * 45)

        if input("\nShow impact graph? (y/n): ").strip().lower() == 'y':
            plot_short_term_impact(selected_brand, "Competitor Action", normal_period_revenue, actual_period_revenue, lost_revenue)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")



# Case 8: Sudden Spike in sales, why?
def run_anomaly_detector():
    print("\n" + "="*50)
    print(" 🚨 SCENARIO: SALES SPIKE ANOMALY DETECTION")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    date_col = 'Date' if 'Date' in df.columns else 'Invoice Date' # Adjust if your date col is named differently
    
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)

    brands = df[brand_col].unique()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to scan for anomalies: ").strip()
        if choice.lower() == 'exit': return
        selected_brand = brands[int(choice) - 1]
        
        brand_df = df[df[brand_col] == selected_brand].copy()
        
        # 1. Calculate Statistical Baselines
        mean_sales = brand_df['Total_Gross_Amount'].mean()
        std_sales = brand_df['Total_Gross_Amount'].std()
        threshold = mean_sales + (1.5 * std_sales) # 1.5 standard deviations is a solid spike
        
        mean_vol = brand_df['Total_Invoices'].mean()
        mean_atv = brand_df['Daily_ATV'].mean()
        
        # 2. Find Anomalies
        spikes = brand_df[brand_df['Total_Gross_Amount'] > threshold]
        
        print("\n" + "-"*50)
        print(f" 📊 ANOMALY REPORT: {selected_brand}")
        print("-" * 50)
        print(f"Normal Daily Average: ₹ {mean_sales:,.2f}")
        print(f"Spike Threshold:      ₹ {threshold:,.2f} (Mean + 1.5 Std Dev)\n")
        
        if spikes.empty:
            print("✅ No massive sudden spikes detected in this period. Sales were highly consistent.")
        else:
            print(f"🚨 Found {len(spikes)} anomaly spike days!\n")
            for index, row in spikes.iterrows():
                date = row[date_col] if date_col in row else "Unknown Date"
                sales = row['Total_Gross_Amount']
                vol = row['Total_Invoices']
                atv = row['Daily_ATV']
                
                vol_change = ((vol - mean_vol) / mean_vol) * 100
                atv_change = ((atv - mean_atv) / mean_atv) * 100
                
                print(f"📅 Date: {date} | Sales: ₹ {sales:,.2f}")
                
                # Determine the primary factor
                if vol_change > atv_change:
                    print(f"   Factor: VOLUME DRIVEN. Invoices were {vol_change:.0f}% higher than normal.")
                else:
                    print(f"   Factor: ATV DRIVEN. Basket sizes were {atv_change:.0f}% larger than normal.")
                print("   ---")

        if input("\nShow scatter plot of anomalies? (y/n): ").strip().lower() == 'y':
            plot_anomalies(selected_brand, brand_df, threshold, mean_sales)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_anomalies(brand, df, threshold, mean):
    plt.figure(figsize=(10, 6))
    
    # Normal days
    normal_df = df[df['Total_Gross_Amount'] <= threshold]
    plt.scatter(range(len(normal_df)), normal_df['Total_Gross_Amount'], color='tab:blue', alpha=0.6, label='Normal Days')
    
    # Spike days
    spike_df = df[df['Total_Gross_Amount'] > threshold]
    spike_indices = df.index[df['Total_Gross_Amount'] > threshold].tolist()
    # Map global indices to local plot indices for x-axis
    local_indices = [df.index.get_loc(idx) for idx in spike_indices]
    
    plt.scatter(local_indices, spike_df['Total_Gross_Amount'], color='tab:red', s=100, edgecolor='black', label='Spike Anomalies')
    
    # Lines
    plt.axhline(mean, color='gray', linestyle='--', label='Average Sales')
    plt.axhline(threshold, color='red', linestyle='-.', alpha=0.5, label='Anomaly Threshold')
    
    plt.title(f"Sales Anomaly Detection: {brand}", fontsize=14, fontweight='bold')
    plt.ylabel("Daily Turnover (₹)")
    plt.xlabel("Timeline (Days)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



# Case 8: Avg sale vs new marketing strategy 
def run_marketing_roi_simulator():
    print("\n" + "="*50)
    print(" 🎯 SCENARIO: MARKETING STRATEGY ROI COMPARISON")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)
    
    baselines = df.groupby(brand_col).agg(
        Median_ATV=('Daily_ATV', 'median'),
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to simulate: ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        
        base_atv = brand_data['Median_ATV']
        base_vol = brand_data['Median_Volume']
        base_daily_turnover = base_atv * base_vol
        
        # Inputs
        campaign_days = int(input("\nEnter duration of the marketing campaign (in days): "))
        campaign_cost = float(input("Enter total cost of the marketing strategy (₹): "))
        expected_lift = float(input("Enter expected % increase in Daily Volume (footfall): "))
        
        # Math
        baseline_revenue = base_daily_turnover * campaign_days
        
        new_vol = base_vol * (1 + (expected_lift / 100))
        new_daily_turnover = new_vol * base_atv
        new_total_revenue = new_daily_turnover * campaign_days
        
        gross_profit_from_ads = new_total_revenue - baseline_revenue
        net_profit_roi = gross_profit_from_ads - campaign_cost
        
        # Output
        print("\n" + "-"*45)
        print(f" 📈 MARKETING ROI ANALYSIS: {selected_brand} ({campaign_days} Days)")
        print("-" * 45)
        print(f"Baseline Expected Sales: ₹ {baseline_revenue:,.2f}")
        print(f"Projected Strategy Sales: ₹ {new_total_revenue:,.2f}")
        print(f"Extra Revenue Generated: +₹ {gross_profit_from_ads:,.2f}")
        print(f"Marketing Cost:          -₹ {campaign_cost:,.2f}")
        print("-" * 45)
        
        if net_profit_roi >= 0:
            print(f"✅ Strategy Outcome: PROFITABLE (Net Gain: ₹ {net_profit_roi:,.2f})")
        else:
            print(f"❌ Strategy Outcome: NET LOSS (Lost: -₹ {abs(net_profit_roi):,.2f})")
            
    except (ValueError, IndexError):
        print("\n[!] Invalid input.")


# Case 9: Sudden spike in number of transactions, why?
def run_transaction_behavior_analyzer():
    print("\n" + "="*50)
    print(" 🛒 SCENARIO: TRANSACTION BEHAVIOR ANALYZER")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    date_col = 'Date' if 'Date' in df.columns else 'Invoice Date'
    
    df['Daily_ATV'] = df.apply(lambda row: row['Total_Gross_Amount'] / row['Total_Invoices'] if row['Total_Invoices'] > 0 else 0, axis=1)

    brands = df[brand_col].unique()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to analyze: ").strip()
        if choice.lower() == 'exit': return
        selected_brand = brands[int(choice) - 1]
        
        brand_df = df[df[brand_col] == selected_brand].copy()
        
        # Baselines
        mean_vol = brand_df['Total_Invoices'].mean()
        std_vol = brand_df['Total_Invoices'].std()
        vol_threshold = mean_vol + (1.5 * std_vol) # Spike threshold for invoices
        
        mean_atv = brand_df['Daily_ATV'].mean()
        
        # Find Spikes
        spikes = brand_df[brand_df['Total_Invoices'] > vol_threshold]
        
        print("\n" + "-"*50)
        print(f" 🔍 BEHAVIOR REPORT: {selected_brand}")
        print("-" * 50)
        print(f"Normal Daily Transactions: {mean_vol:.0f} invoices")
        print(f"Spike Threshold:           {vol_threshold:.0f} invoices\n")
        
        if spikes.empty:
            print("✅ No sudden transaction spikes detected. Footfall is stable.")
        else:
            print(f"🚨 Found {len(spikes)} days with unusual transaction volume!\n")
            for index, row in spikes.iterrows():
                date = row[date_col] if date_col in row else "Unknown Date"
                vol = row['Total_Invoices']
                atv = row['Daily_ATV']
                
                print(f"📅 Date: {date} | Transactions: {vol:.0f} (Normal: {mean_vol:.0f})")
                
                # Analyze Behavior based on ATV
                if atv < (mean_atv * 0.8): # ATV dropped by more than 20%
                    print(f"   🧠 Behavior Insight: BARGAIN HUNTING.")
                    print(f"   Average spend dropped to ₹{atv:,.0f} (Normal: ₹{mean_atv:,.0f}).")
                    print(f"   Customers bought high volume but cheaper items (likely clearance/sale).")
                elif atv > (mean_atv * 1.2): # ATV jumped by more than 20%
                    print(f"   🧠 Behavior Insight: HIGH-INTENT SURGE.")
                    print(f"   Average spend jumped to ₹{atv:,.0f} (Normal: ₹{mean_atv:,.0f}).")
                    print(f"   Customers arrived in large numbers AND bought premium/multiple items.")
                else:
                    print(f"   🧠 Behavior Insight: STANDARD FOOTFALL SURGE.")
                    print(f"   Average spend stayed normal (₹{atv:,.0f}). It was just a very busy day.")
                print("   ---")

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")


# Case 9: Market trend shift
def run_market_trend_simulator():
    print("\n" + "="*50)
    print(" 📈 SCENARIO: MARKET TREND SHIFT SIMULATOR")
    print("="*50)

    file_path = input(r"Enter the exact path to your Daily_Sales_Report.csv: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    
    if brand_col not in df.columns or 'Total_Invoices' not in df.columns:
        print("[!] Error: Missing required columns.")
        return

    baselines = df.groupby(brand_col).agg(
        Median_Volume=('Total_Invoices', 'median')
    ).reset_index()

    brands = baselines[brand_col].tolist()
    print("\nAvailable Brands:")
    for i, brand in enumerate(brands, 1):
        print(f"{i}. {brand}")
        
    try:
        choice = input("\nEnter brand number to simulate: ").strip()
        if choice.lower() == 'exit': return
        
        brand_data = baselines.iloc[int(choice) - 1]
        selected_brand = brand_data[brand_col]
        
        base_monthly_vol = brand_data['Median_Volume'] * 30
        
        print(f"\nBaseline Monthly Transactions: {base_monthly_vol:,.0f}")
        
        # Inputs
        trend_shift = float(input("Enter Expected Monthly Trend Shift % (Use - for declining, + for booming): "))
        months = int(input("How many months to forecast? (e.g., 6): "))
        
        # Math & Data Collection
        current_vol = base_monthly_vol
        projection_data = []
        
        print("\n" + "-"*45)
        print(f" 📊 TREND PROJECTION: {selected_brand}")
        print("-" * 45)
        
        for month in range(1, months + 1):
            current_vol = current_vol * (1 + (trend_shift / 100))
            projection_data.append({'Month': f"M{month}", 'Volume': current_vol})
            print(f"Month {month:02d}: {current_vol:,.0f} projected transactions")
            
        total_change = current_vol - base_monthly_vol
        
        print("-" * 45)
        if total_change > 0:
            print(f"Final Outcome: Gained {total_change:,.0f} monthly transactions.")
        else:
            print(f"Final Outcome: Lost {abs(total_change):,.0f} monthly transactions.")

        if input("\nShow trend graph? (y/n): ").strip().lower() == 'y':
            plot_market_trend(selected_brand, base_monthly_vol, projection_data, trend_shift)
            
    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_market_trend(brand, base_vol, data, shift_pct):
    months = [d['Month'] for d in data]
    volumes = [d['Volume'] for d in data]
    
    plt.figure(figsize=(10, 6))
    
    # Set color based on positive or negative trend
    trend_color = 'tab:green' if shift_pct > 0 else 'tab:red'
    
    # Plot Lines
    plt.plot(months, volumes, color=trend_color, marker='o', linewidth=2.5, label='Projected Trend')
    plt.plot(months, [base_vol]*len(months), color='gray', linestyle='--', linewidth=2, label='Baseline Volume')
    
    # Fill the gap
    plt.fill_between(months, [base_vol]*len(months), volumes, color=trend_color, alpha=0.15)
    
    # Labels & Styling
    plt.title(f"Market Trend Shift ({shift_pct}% Monthly): {brand}", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline (Months)")
    plt.ylabel("Monthly Transactions")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Summary Box
    final_vol = volumes[-1]
    net_change = final_vol - base_vol
    sign = "+" if net_change > 0 else ""
    plt.text(0.05, 0.95, f"Net Transaction Change:\n{sign}{net_change:,.0f} by {months[-1]}", 
             transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
             
    plt.tight_layout()
    plt.show()



# Case 11: Category growth
def run_category_growth_simulator():
    print("\n" + "="*50)
    print(" 📊 SCENARIO: CATEGORY GROWTH IMPACT")
    print("="*50)

    file_path = input(r"Enter the exact path to your monthly CSV data: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    cat_col = 'Category' 
    
    if cat_col not in df.columns or 'Total_Gross_Amount' not in df.columns:
        print(f"[!] Error: Missing required columns (Category or Total_Gross_Amount).")
        return
        
    # Find the Baseline Monthly Turnover directly from the monthly totals
    brand_baselines = df.groupby([cat_col, brand_col]).agg(
        Monthly_Turnover=('Total_Gross_Amount', 'median')
    ).reset_index()

    # Roll up to Category Level
    cat_baselines = brand_baselines.groupby(cat_col)['Monthly_Turnover'].sum().reset_index()
    total_portfolio_revenue = cat_baselines['Monthly_Turnover'].sum()

    categories = cat_baselines[cat_col].tolist()
    print("\nAvailable Categories:")
    for i, cat in enumerate(categories, 1):
        print(f"{i}. {cat}")
        
    try:
        choice = input("\nEnter category number to simulate growth: ").strip()
        if choice.lower() == 'exit': return
        
        selected_cat = categories[int(choice) - 1]
        base_cat_revenue = cat_baselines.loc[cat_baselines[cat_col] == selected_cat, 'Monthly_Turnover'].values[0]
        
        print(f"\n--- Baseline for {selected_cat} ---")
        print(f"Category Monthly Revenue:   ₹ {base_cat_revenue:,.2f}")
        print(f"Total Mall Monthly Revenue: ₹ {total_portfolio_revenue:,.2f}")
        
        # Inputs
        growth_pct = float(input(f"\nEnter Expected % Growth for {selected_cat}: "))
        
        # Math
        new_cat_revenue = base_cat_revenue * (1 + (growth_pct / 100))
        extra_revenue_generated = new_cat_revenue - base_cat_revenue
        new_portfolio_revenue = total_portfolio_revenue + extra_revenue_generated
        overall_mall_growth = (extra_revenue_generated / total_portfolio_revenue) * 100
        
        # Output
        print("\n" + "-"*45)
        print(f" 📈 CATEGORY IMPACT: {selected_cat} (+{growth_pct}%)")
        print("-" * 45)
        print(f"New Category Revenue: ₹ {new_cat_revenue:,.2f}")
        print(f"Extra Cash Generated: +₹ {extra_revenue_generated:,.2f}")
        print("-" * 45)
        print(f"New Total Mall Revenue: ₹ {new_portfolio_revenue:,.2f}")
        print(f"Overall Mall Lift:      +{overall_mall_growth:.2f}%")
        print("-" * 45)

        if input("\nShow impact graph? (y/n): ").strip().lower() == 'y':
            plot_category_growth(selected_cat, base_cat_revenue, new_cat_revenue, growth_pct)

    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_category_growth(category, base_rev, new_rev, growth_pct):
    plt.figure(figsize=(8, 6))
    
    labels = ['Baseline Revenue', 'Projected Revenue']
    values = [base_rev, new_rev]
    colors = ['tab:gray', 'tab:green']
    
    bars = plt.bar(labels, values, color=colors, width=0.5)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval * 1.01, 
                 f"₹ {yval/100000:,.1f}L", ha='center', va='bottom', fontweight='bold')
    
    plt.title(f"Category Growth: {category} (+{growth_pct}%)", fontsize=14, fontweight='bold')
    plt.ylabel("Monthly Revenue (₹)")
    
    extra = new_rev - base_rev
    plt.text(0.5, max(values) * 1.1, f"Net Gain:\n+₹ {extra:,.2f}", 
             ha='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='green'))
             
    ymin, ymax = plt.ylim()
    plt.ylim(0, ymax * 1.25)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# Case 11: New product in category
def run_category_product_launch_simulator():
    print("\n" + "="*50)
    print(" 🚀 SCENARIO: NEW PRODUCT LAUNCH (CATEGORY LEVEL)")
    print("="*50)

    file_path = input(r"Enter the exact path to your monthly CSV data: ").strip()
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("[!] Error: File not found.")
        return

    brand_col = 'Tenant Name' if 'Tenant Name' in df.columns else 'Brand Name'
    cat_col = 'Category' 
    
    if cat_col not in df.columns:
        print(f"[!] Error: '{cat_col}' column not found in your CSV.")
        return
        
    # Baseline calculations
    brand_baselines = df.groupby([cat_col, brand_col]).agg(
        Monthly_Turnover=('Total_Gross_Amount', 'median')
    ).reset_index()

    # Roll up to Category Level
    cat_monthly_baselines = brand_baselines.groupby(cat_col)['Monthly_Turnover'].sum().reset_index()
    
    # Sort to show high-performing categories at the top
    cat_monthly_baselines = cat_monthly_baselines.sort_values(by='Monthly_Turnover', ascending=False).reset_index(drop=True)

    print("\nTop Performing Categories (By Baseline Monthly Turnover):")
    for i, row in cat_monthly_baselines.iterrows():
        print(f"{i+1}. {row[cat_col]} (₹ {row['Monthly_Turnover']:,.0f} / month)")
        
    try:
        choice = input("\nEnter category number launching the product: ").strip()
        if choice.lower() == 'exit': return
        
        selected_cat = cat_monthly_baselines.iloc[int(choice) - 1][cat_col]
        base_cat_monthly_turnover = cat_monthly_baselines.iloc[int(choice) - 1]['Monthly_Turnover']
        
        # Derive a clean daily turnover from the monthly median to use for the launch days math
        base_cat_daily_turnover = base_cat_monthly_turnover / 30
        
        # Inputs
        launch_days = int(input(f"\nEnter duration of launch hype/campaign in {selected_cat} (in days): "))
        product_price = float(input("Enter the exact price of the NEW product (₹): "))
        expected_daily_units = int(input(f"Enter expected units sold per day across {selected_cat}: "))
        
        # Math
        normal_period_revenue = base_cat_daily_turnover * launch_days
        launch_extra_revenue = product_price * expected_daily_units * launch_days
        total_projected_revenue = normal_period_revenue + launch_extra_revenue
        lift_percentage = (launch_extra_revenue / normal_period_revenue) * 100
        
        # Output
        print("\n" + "-"*45)
        print(f" 🌟 CATEGORY LAUNCH PROJECTION: {selected_cat} ({launch_days} Days)")
        print("-" * 45)
        print(f"Normal Expected Category Revenue:  ₹ {normal_period_revenue:,.2f}")
        print(f"Revenue from New Product:          +₹ {launch_extra_revenue:,.2f}")
        print("-" * 45)
        print(f"Total Projected Category Revenue:  ₹ {total_projected_revenue:,.2f}")
        print(f"Category Period Lift:              +{lift_percentage:.1f}%")
        print("-" * 45)

        if input("\nShow launch graph? (y/n): ").strip().lower() == 'y':
            plot_category_product_launch(selected_cat, normal_period_revenue, launch_extra_revenue, launch_days)
        
    except (ValueError, IndexError):
        print("\n[!] Invalid input.")

def plot_category_product_launch(category, normal_rev, extra_rev, days):
    plt.figure(figsize=(7, 7))
    
    # Stacked Bar Chart
    bar_width = 0.6
    plt.bar(["Projected Period"], [normal_rev], color='tab:purple', width=bar_width, label="Normal Category Revenue")
    plt.bar(["Projected Period"], [extra_rev], bottom=[normal_rev], color='tab:orange', width=bar_width, label="New Product Revenue")
    
    # Add text labels inside the bars
    plt.text(0, normal_rev/2, f"Baseline\n₹ {normal_rev/100000:,.1f}L", ha='center', va='center', color='white', fontweight='bold')
    plt.text(0, normal_rev + (extra_rev/2), f"+ ₹ {extra_rev/100000:,.1f}L\n(Launch)", ha='center', va='center', color='black', fontweight='bold')
    
    plt.title(f"New Product Launch Impact on Category: {category} ({days} Days)", fontsize=14, fontweight='bold')
    plt.ylabel("Total Revenue (₹)")
    plt.legend(loc="upper left")
    
    # Total Box
    total = normal_rev + extra_rev
    plt.text(1.1, total, f"Total Captured:\n₹ {total:,.2f}", 
             ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange'))
             
    plt.xlim(-0.5, 1.5)
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
# For 366 days data functions
    # combine_master_data() 
    # calculate_monthly_sales_with_mall()
    # filter_csv_by_mall()

# For invoices files functions
    # combine_invoice_data()
    # calculate_daily_sales_invoice()
    # calculate_monthly_sales_invoice() 

# Scenarios functions

    # Case 1
    # run_price_simulator()   #price increment 10% logic

    # Case 2
    # run_online_shift_simulator()  #online preference shift

    # Case 6
    # run_trend_projection_simulator()  #increase in shopping trend 90 days
    # run_macro_spending_simulator()  #consumer spending habit

    # Case 7
    # run_supply_chain_simulator() #Supply Chain
    # run_competitor_action_simulator() #Competitor lowered their price

    # Case 8
    # run_anomaly_detector()  #Sudden spike in sale
    # run_marketing_roi_simulator() #Avg sale vs new marketing strategy

    # Case 9
    # run_transaction_behavior_analyzer() #Increase in number of transactions
    # run_market_trend_simulator() #Market trend shift


    # Case 11
    run_category_growth_simulator() # Category growth
    run_category_product_launch_simulator() # New product in category