import nselib.capital_market
import nselib.derivatives
import pandas as pd
from flask import Flask, render_template, request, jsonify
import sqlite3
from datetime import datetime, timedelta
import time
import os

app = Flask(__name__)
DATABASE = 'nselib_data.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    print("Attempting to initialize database...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS option_data (
                    unique_id TEXT UNIQUE,
                    TIMESTAMP TEXT,
                    INSTRUMENT TEXT,
                    SYMBOL TEXT,
                    EXPIRY_DT TEXT,
                    STRIKE_PRICE REAL,
                    OPTION_TYPE TEXT,
                    OPENING_PRICE REAL,
                    TRADE_HIGH_PRICE REAL,
                    TRADE_LOW_PRICE REAL,
                    CLOSING_PRICE REAL,
                    PREV_CLS REAL,
                    TOT_TRADED_QTY REAL,
                    TOT_TRADED_VAL REAL,
                    OPEN_INT REAL,
                    CHANGE_IN_OI REAL,
                    MARKET_LOT REAL,
                    UNDERLYING_VALUE REAL,
                    MARKET_TYPE TEXT
            );
            CREATE TABLE IF NOT EXISTS fno_stocks (
                underlying TEXT,
                symbol TEXT UNIQUE,
                serialNumber TEXT,
                Preferred_Stock BOOLEAN,
                Last_Updated TEXT,
                status TEXT DEFAULT 'active'
            );
            ALTER TABLE fno_stocks ADD COLUMN status TEXT DEFAULT 'active';
        ''')
            conn.commit()
            print("Database initialized successfully.")
        except Exception as e:
            print(f"Error initializing database: {e}")

def ensure_settings_table_exists():
    print("Ensuring settings table exists...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );
            ''')
            # Default values for all settings
            default_settings = {
                'sleep_duration': '15',
                'margin_percentage': '50',
                'transaction_cost': '0.1',
                'stcg_percentage': '15',
                'min_interest_percentage': '2.0'  # New default for Covered Call
            }
            for key, value in default_settings.items():
                cursor.execute("INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)", (key, value))
            
            conn.commit()
            print("Settings table ensured and defaults are set.")
        except Exception as e:
            print(f"Error ensuring settings table: {e}")

@app.before_request
def before_request():
    ensure_settings_table_exists()

@app.route('/api/settings')
def get_all_settings():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM settings")
        settings_data = {row['key']: row['value'] for row in cursor.fetchall()}
        conn.close()
        return jsonify({"status": "success", "settings": settings_data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/settings/save', methods=['POST'])
def save_all_settings():
    try:
        data = request.get_json()
        
        # Basic validation for all settings
        required_keys = ['sleep_duration', 'margin_percentage', 'transaction_cost', 'stcg_percentage', 'min_interest_percentage']
        if not all(k in data and isinstance(data[k], (int, float)) for k in required_keys):
            return jsonify({"status": "error", "message": "Invalid data format. All values must be numbers."}), 400

        if data['sleep_duration'] < 15:
            return jsonify({"status": "error", "message": "Sleep duration cannot be less than 15 seconds."}), 400
        
        if data['min_interest_percentage'] < 0:
            return jsonify({"status": "error", "message": "Minimum Interest Percentage cannot be negative."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        for key, value in data.items():
            cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str(value)))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "All settings saved successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def get_db_columns():
    print("Attempting to get database columns...")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(option_data);")
            columns = [row[1] for row in cursor.fetchall()]
            print(f"Retrieved columns: {columns}")
            return columns
        except Exception as e:
            print(f"Error getting database columns: {e}")
            return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_data')
def fetch_data_page():
    return render_template('fetch_data.html')



@app.route('/insights')
def insights_page():
    return render_template('insights.html')

@app.route('/settings')
def settings_page():
    return render_template('settings.html')

def get_db_columns():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(option_data);")
        columns = [row[1] for row in cursor.fetchall()]
        return columns

@app.route('/fetch_fno_stocks', methods=['POST'])
def fetch_fno_stocks_data():
    try:
        fetched_stocks_df = nselib.capital_market.fno_equity_list()

        # Guardrail: Check if the fetched list has at least 100 stocks
        if fetched_stocks_df.empty or len(fetched_stocks_df) < 100:
            return jsonify({"status": "error", "message": f"Guardrail triggered: Fetched only {len(fetched_stocks_df)} F&O stocks. Not updating database to prevent accidental deactivation."})

        conn = get_db_connection()
        cursor = conn.cursor()

        # Step 1: Mark all existing stocks as 'inactive'
        cursor.execute("UPDATE fno_stocks SET status = 'inactive'")
        conn.commit()

        # Step 2: Process newly fetched stocks
        for index, row in fetched_stocks_df.iterrows():
            symbol = row['symbol']
            underlying = row['underlying']
            serial_number = row['serialNumber']
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Check if stock already exists
            cursor.execute("SELECT COUNT(*) FROM fno_stocks WHERE symbol = ?", (symbol,))
            exists = cursor.fetchone()[0]

            if exists:
                # Update existing stock to 'active'
                cursor.execute(
                    "UPDATE fno_stocks SET underlying = ?, serialNumber = ?, Last_Updated = ?, status = 'active' WHERE symbol = ?",
                    (underlying, serial_number, current_time, symbol)
                )
            else:
                # Insert new stock as 'active'
                cursor.execute(
                    "INSERT INTO fno_stocks (underlying, symbol, serialNumber, Preferred_Stock, Last_Updated, status) VALUES (?, ?, ?, ?, ?, ?)",
                    (underlying, symbol, serial_number, False, current_time, 'active')
                )
            conn.commit()

        conn.close()
        return jsonify({"status": "success", "message": "F&O stocks list updated successfully."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching and updating F&O stocks: {e}"})

@app.route('/fetch_custom_data')
def fetch_custom_data_page():
    return render_template('fetch_custom_data.html')

@app.route('/api/settings/sleep_duration')
def get_sleep_duration():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = 'sleep_duration'")
        sleep_duration = cursor.fetchone()
        conn.close()
        if sleep_duration:
            return jsonify({"status": "success", "sleep_duration": int(sleep_duration[0])})
        else:
            return jsonify({"status": "error", "message": "Sleep duration not found."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/settings/save_sleep_duration', methods=['POST'])
def save_sleep_duration():
    try:
        data = request.get_json()
        sleep_duration = data.get('sleep_duration')

        if sleep_duration is None or not isinstance(sleep_duration, int) or sleep_duration < 15:
            return jsonify({"status": "error", "message": "Invalid sleep duration. Must be an integer and at least 15."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ('sleep_duration', str(sleep_duration)))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": "Sleep duration saved successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/get_fno_symbols')
def get_fno_symbols():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT symbol FROM fno_stocks WHERE status = 'active' ORDER BY symbol")
        symbols = [row['symbol'] for row in cursor.fetchall()]
        conn.close()
        return jsonify(symbols)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/fetch_bulk_data')
def fetch_bulk_data_page():
    return render_template('fetch_bulk_data.html')

@app.route('/view_fno_stocks')
def view_fno_stocks():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT underlying, symbol, serialNumber, Preferred_Stock, Last_Updated, status FROM fno_stocks ORDER BY symbol")
        fno_stocks_data = cursor.fetchall()
        conn.close()
        return render_template('view_fno_stocks.html', fno_stocks=fno_stocks_data)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/update_preferred_stock', methods=['POST'])
def update_preferred_stock():
    try:
        data = request.get_json()
        symbol = data.get('symbol')
        preferred = data.get('preferred')

        if symbol is None or preferred is None:
            return jsonify({"status": "error", "message": "Missing symbol or preferred status."}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        # SQLite stores BOOLEAN as INTEGER (0 for False, 1 for True)
        preferred_value = 1 if preferred else 0
        cursor.execute("UPDATE fno_stocks SET Preferred_Stock = ? WHERE symbol = ?", (preferred_value, symbol))
        conn.commit()
        conn.close()
        return jsonify({"status": "success", "message": f"Preferred status for {symbol} updated to {preferred}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/fetch_custom_option_data', methods=['POST'])
def fetch_custom_option_data():
    try:
        data = request.get_json()
        selected_symbols = data.get('stocks', [])
        instrument = data.get('instrument')
        option_type_selection = data.get('option_type')
        from_date_str = data.get('from_date')
        to_date_str = data.get('to_date')
        period = data.get('period')

        # Input Validation
        if not all([selected_symbols, instrument, option_type_selection, from_date_str, to_date_str, period]):
            return jsonify({"status": "error", "message": "Missing required parameters."}), 400

        try:
            from_date = datetime.strptime(from_date_str, '%Y-%m-%d').strftime('%d-%m-%Y')
            to_date = datetime.strptime(to_date_str, '%Y-%m-%d').strftime('%d-%m-%Y')
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid date format. Use YYYY-MM-DD."}), 400

        db_columns = get_db_columns()
        conn = get_db_connection()
        total_fetched_records = 0

        # Get sleep duration from settings
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = 'sleep_duration'")
        sleep_duration_setting = cursor.fetchone()
        sleep_duration = int(sleep_duration_setting[0]) if sleep_duration_setting else 15 # Default to 15 if not found

        option_types_to_fetch = []
        if option_type_selection == 'Both':
            option_types_to_fetch = ['CE', 'PE']
        else:
            option_types_to_fetch = [option_type_selection]

        for symbol in selected_symbols:
            for opt_type in option_types_to_fetch:
                try:
                    print(f"Fetching custom data for {symbol}, {opt_type} from {from_date} to {to_date}")
                    fetched_data = nselib.derivatives.option_price_volume_data(
                        symbol=symbol,
                        instrument=instrument,
                        option_type=opt_type,
                        from_date=from_date,
                        to_date=to_date,
                        period=period
                    )

                    if not fetched_data.empty:
                        # Standardize data types before creating unique_id
                        fetched_data['SYMBOL'] = fetched_data['SYMBOL'].astype(str).str.strip()
                        fetched_data['EXPIRY_DT'] = fetched_data['EXPIRY_DT'].astype(str).str.strip()
                        fetched_data['STRIKE_PRICE'] = fetched_data['STRIKE_PRICE'].astype(float).astype(str)
                        fetched_data['OPTION_TYPE'] = fetched_data['OPTION_TYPE'].astype(str).str.strip()

                        # Convert TIMESTAMP to datetime objects and then to string for SQLite
                        fetched_data['TIMESTAMP'] = pd.to_datetime(fetched_data['TIMESTAMP']).dt.strftime('%Y-%m-%d %H:%M:%S')

                        # Calculate unique_id
                        fetched_data['unique_id'] = (
                            fetched_data['SYMBOL'] + '_' + 
                            fetched_data['EXPIRY_DT'] + '_' + 
                            fetched_data['STRIKE_PRICE'] + '_' + 
                            fetched_data['OPTION_TYPE'] + '_' + 
                            fetched_data['TIMESTAMP']
                        )

                        # Ensure TOT_TRADED_QTY is numeric before filtering
                        fetched_data['TOT_TRADED_QTY'] = pd.to_numeric(fetched_data['TOT_TRADED_QTY'], errors='coerce').fillna(0)
                        
                        # Filter rows where TOT_TRADED_QTY < 100
                        fetched_data = fetched_data[fetched_data['TOT_TRADED_QTY'] >= 100]

                        if not fetched_data.empty:
                            # Ensure we only try to insert columns that exist in the DB
                            db_cols = get_db_columns()
                            filtered_cols = [col for col in fetched_data.columns if col in db_cols]
                            fetched_data = fetched_data[filtered_cols]

                            # Use INSERT OR IGNORE to handle duplicates gracefully
                            cursor = conn.cursor()
                            cols_str = ', '.join(fetched_data.columns)
                            placeholders = ', '.join('?' * len(fetched_data.columns))
                            sql = f"INSERT OR IGNORE INTO option_data ({cols_str}) VALUES ({placeholders})"
                            
                            inserted_count = 0
                            for row in fetched_data.itertuples(index=False, name=None):
                                cursor.execute(sql, row)
                                if cursor.rowcount > 0:
                                    inserted_count += 1
                            
                            conn.commit()
                            ignored_count = len(fetched_data) - inserted_count
                            total_fetched_records += inserted_count
                            print(f"For {symbol} ({opt_type}): Stored {inserted_count} new records, ignored {ignored_count} duplicate records.")
                    time.sleep(sleep_duration) # Decent space between API calls
                except Exception as e:
                    print(f"Error fetching custom data for {symbol}, {opt_type}: {e}")
                    # Continue to next stock/option type even if one fails
                    time.sleep(sleep_duration) # Still wait to avoid blocking even on error

        conn.close()
        return jsonify({"status": "success", "message": f"Custom data fetching complete. Total records stored: {total_fetched_records}."})

    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {e}"}), 500



@app.route('/view_data')
def view_data():
    page = request.args.get('page', 1, type=int)
    per_page = 100
    offset = (page - 1) * per_page
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM option_data LIMIT {per_page} OFFSET {offset}", conn)
    total_rows = conn.execute('SELECT COUNT(*) FROM option_data').fetchone()[0]
    conn.close()
    total_pages = (total_rows + per_page - 1) // per_page
    return render_template('view_data.html', tables=[df.to_html(classes='data')], titles=df.columns.values, page=page, total_pages=total_pages)

@app.route('/cleanup_data')
def cleanup_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM option_data WHERE TOT_TRADED_QTY < 100")
        conn.commit()
        rows_deleted = cursor.rowcount
        conn.close()
        return jsonify({"status": "success", "message": f"Deleted {rows_deleted} rows from option_data."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analyze/covered_call')
def covered_call_page():
    return render_template('covered_call.html')

@app.route('/api/analyze/covered_call_data')
def analyze_covered_call_data():
    try:
        conn = get_db_connection()
        top_n = request.args.get('top_n', type=int)
        stock_filter = request.args.get('stock_filter', 'all') # Default to 'all'

        # Find the most recent timestamp
        most_recent_timestamp = pd.read_sql_query("SELECT MAX(TIMESTAMP) FROM option_data", conn).iloc[0, 0]
        if not most_recent_timestamp:
            return jsonify({"status": "error", "message": "No data found in option_data table."})

        # Get the current month and year for expiry date filtering
        # current_month = datetime.now().strftime('%b').upper()
        # current_year = str(datetime.now().year)

        # Build the base query
        query = "SELECT * FROM option_data WHERE TIMESTAMP = ?"
        params = [most_recent_timestamp]

        if stock_filter == 'preferred':
            # Fetch preferred stocks
            preferred_stocks_df = pd.read_sql_query("SELECT symbol FROM fno_stocks WHERE Preferred_Stock = 1", conn)
            if preferred_stocks_df.empty:
                return jsonify({"status": "error", "message": "No preferred stocks found."})
            preferred_symbols = preferred_stocks_df['symbol'].tolist()
            
            # Add symbol filter to the query
            placeholders = ','.join('?' * len(preferred_symbols))
            query += f" AND SYMBOL IN ({placeholders})"
            params.extend(preferred_symbols)

        df = pd.read_sql_query(query, conn, params=params)
        print(f"DEBUG: analyze_covered_call_data - TIMESTAMP column dtype: {df['TIMESTAMP'].dtype}")
        print(f"DEBUG: analyze_covered_call_data - Sample TIMESTAMP values: {df['TIMESTAMP'].head().tolist()}")

        if df.empty:
            return jsonify({"status": "error", "message": "No data for the most recent timestamp or selected stocks."})

        # Filter for current month's expiry
        # df = df[df['EXPIRY_DT'].str.contains(current_month, case=False) & df['EXPIRY_DT'].str.contains(current_year)]

        # if df.empty:
        #     return jsonify({"status": "error", "message": "No data for the current month's expiry or selected stocks."})

        # Ensure columns are numeric for the calculation
        df['CLOSING_PRICE'] = pd.to_numeric(df['CLOSING_PRICE'], errors='coerce')
        df['UNDERLYING_VALUE'] = pd.to_numeric(df['UNDERLYING_VALUE'], errors='coerce')
        df['STRIKE_PRICE'] = pd.to_numeric(df['STRIKE_PRICE'], errors='coerce')
        df['OPEN_INT'] = pd.to_numeric(df['OPEN_INT'], errors='coerce')
        df.dropna(subset=['CLOSING_PRICE', 'UNDERLYING_VALUE', 'STRIKE_PRICE', 'OPEN_INT'], inplace=True)

        # Covered Call A+B strategy logic
        # Get minimum interest percentage from settings
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = 'min_interest_percentage'")
        min_interest_setting = cursor.fetchone()
        min_interest_percentage = float(min_interest_setting[0]) / 100 if min_interest_setting else 0.02 # Default to 0.02 if not found
        print(f"DEBUG: Using min_interest_percentage: {min_interest_percentage}")

        df = df[df['UNDERLYING_VALUE'] != 0]
        df_filtered = df[(df['CLOSING_PRICE'] / df['UNDERLYING_VALUE'] > min_interest_percentage) & (df['OPTION_TYPE'] == 'CE')]
        df_filtered = df_filtered[df_filtered['STRIKE_PRICE'] >= df_filtered['UNDERLYING_VALUE']]

        # Add calculated columns for Covered Call A+B
        df_filtered['Interest'] = df_filtered.apply(lambda row: round((row['CLOSING_PRICE'] * 100 / row['UNDERLYING_VALUE']), 1) if row['UNDERLYING_VALUE'] != 0 else 0.0, axis=1)
        df_filtered['Appreciation'] = df_filtered.apply(lambda row: round(((row['STRIKE_PRICE'] - row['UNDERLYING_VALUE']) * 100 / row['UNDERLYING_VALUE']), 1) if row['UNDERLYING_VALUE'] != 0 else 0.0, axis=1)
        df_filtered['Return'] = df_filtered.apply(lambda row: round((row['Interest'] + row['Appreciation']), 1), axis=1)

        # Apply top N filter after strategy-specific filtering
        if top_n is not None and top_n > 0:
            df_filtered = df_filtered.groupby('SYMBOL').apply(lambda x: x.nlargest(top_n, 'OPEN_INT')).reset_index(drop=True)

        conn.close()

        if df_filtered.empty:
            return jsonify({"status": "success", "message": "No results found for the given strategy.", "data": []})

        return jsonify({"status": "success", "data": df_filtered.to_dict(orient='records')})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500




@app.route('/api/get_analysis_symbols')
def get_analysis_symbols():
    try:
        conn = get_db_connection()
        # Query for distinct symbols that have data in the option_data table
        symbols = pd.read_sql_query("SELECT DISTINCT SYMBOL FROM option_data ORDER BY SYMBOL", conn)
        conn.close()
        return jsonify(symbols['SYMBOL'].tolist())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/analyze/oi_shift')
def oi_shift_page():
    return render_template('oi_shift.html')

@app.route('/api/get_expiry_dates')
def get_expiry_dates():
    try:
        conn = get_db_connection()
        # Query for distinct expiry dates from the option_data table
        # Order by date to ensure current month is likely first
        expiry_dates_df = pd.read_sql_query("SELECT DISTINCT EXPIRY_DT FROM option_data ORDER BY EXPIRY_DT ASC", conn)
        conn.close()
        return jsonify(expiry_dates_df['EXPIRY_DT'].tolist())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/analyze/oi_shift_data', methods=['POST'])
def analyze_oi_shift_data():
    try:
        conn = get_db_connection()
        top_strike = 5
        
        # Get selected stocks, display type, and expiry from the POST request
        data = request.get_json()
        selected_symbols = data.get('stocks', [])
        display_type = data.get('display_type', 'percentage') # Default to 'percentage'
        selected_expiry = data.get('expiry') # New: Get selected expiry

        # Step 1 & 2: Initial lightweight query to find top strikes
        print("DEBUG: Starting lightweight query to identify top strikes.")
        
        # Base query
        query_sql = "SELECT SYMBOL, TIMESTAMP, STRIKE_PRICE, OPEN_INT, TOT_TRADED_QTY, EXPIRY_DT FROM option_data"
        params = []
        where_clauses = []

        # Add expiry filter if selected
        if selected_expiry:
            where_clauses.append("EXPIRY_DT = ?")
            params.append(selected_expiry)

        # If specific stocks are selected (and not 'All'), add a WHERE clause
        if selected_symbols:
            placeholders = ', '.join('?' for symbol in selected_symbols)
            where_clauses.append(f"SYMBOL IN ({placeholders})")
            params.extend(selected_symbols)

        if where_clauses:
            query_sql += " WHERE " + " AND ".join(where_clauses)

        pre_df = pd.read_sql_query(query_sql, conn, params=params)

        if pre_df.empty:
            conn.close()
            return jsonify({"status": "error", "message": "No data found for the selected symbols and expiry."})

        pre_df['TIMESTAMP'] = pd.to_datetime(pre_df['TIMESTAMP'])
        pre_df['OPEN_INT'] = pd.to_numeric(pre_df['OPEN_INT'], errors='coerce').fillna(0)
        pre_df['TOT_TRADED_QTY'] = pd.to_numeric(pre_df['TOT_TRADED_QTY'], errors='coerce').fillna(0)


        # Find the latest timestamp for each symbol
        latest_timestamps = pre_df.groupby('SYMBOL')['TIMESTAMP'].max().reset_index()
        latest_data = pd.merge(pre_df, latest_timestamps, on=['SYMBOL', 'TIMESTAMP'])

        # Find the top N strike prices for the latest timestamp for each symbol
        top_strikes_df = latest_data.sort_values('OPEN_INT', ascending=False).groupby('SYMBOL').head(top_strike)
        
        if top_strikes_df.empty:
            conn.close()
            return jsonify({"status": "success", "message": "No top strikes found to analyze for the selection.", "data": []})

        # Create a set of unique (symbol, strike_price) tuples for the final query
        unique_strikes_to_query = set(zip(top_strikes_df['SYMBOL'], top_strikes_df['STRIKE_PRICE']))

        # Build the WHERE clause for the main query
        conditions = []
        query_params = []
        for symbol, strike in unique_strikes_to_query:
            conditions.append("(SYMBOL = ? AND STRIKE_PRICE = ?)")
            query_params.extend([symbol, float(strike)])

        where_clause = " OR ".join(conditions)
        
        # Step 3: Targeted final query
        print(f"DEBUG: Executing targeted query for {len(unique_strikes_to_query)} unique symbol/strike pairs.")
        main_query = f"SELECT * FROM option_data WHERE {where_clause}"
        df = pd.read_sql_query(main_query, conn, params=query_params)
        conn.close()

        print(f"DEBUG: Main DataFrame shape after targeted query: {df.shape}")
        if df.empty:
            return jsonify({"status": "error", "message": "Could not fetch detailed data for the top strike prices."})

        # --- Start of existing analysis logic ---
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
        df['OPEN_INT'] = pd.to_numeric(df['OPEN_INT'], errors='coerce').fillna(0)
        df['STRIKE_PRICE'] = pd.to_numeric(df['STRIKE_PRICE'], errors='coerce').fillna(0)
        df['TOT_TRADED_QTY'] = pd.to_numeric(df['TOT_TRADED_QTY'], errors='coerce').fillna(0)
        df.dropna(subset=['OPEN_INT', 'STRIKE_PRICE', 'TOT_TRADED_QTY'], inplace=True)

        all_stock_tables = []
        # Use the originally selected symbols to maintain order and selection
        symbols_to_process = selected_symbols if selected_symbols else df['SYMBOL'].unique()

        for symbol in symbols_to_process:
            symbol_df = df[df['SYMBOL'] == symbol].copy()
            if symbol_df.empty:
                continue

            # Get the expiry date for the current symbol
            # Assuming EXPIRY_DT is consistent for a given symbol in the filtered data
            expiry_dt_str = symbol_df['EXPIRY_DT'].iloc[0]
            # Parse the expiry date string to get the day
            # Example: '27-JUN-2024' -> 27
            expiry_day = datetime.strptime(expiry_dt_str, '%d-%b-%Y').day
            print(f"DEBUG: Symbol: {symbol}, Expiry DT: {expiry_dt_str}, Expiry Day: {expiry_day}")

            # Separate CE and PE data
            filtered_by_strike_ce = symbol_df[symbol_df['OPTION_TYPE'] == 'CE'].copy()
            filtered_by_strike_pe = symbol_df[symbol_df['OPTION_TYPE'] == 'PE'].copy()

            # Process CE data
            oi_data_ce, volume_data_ce = {}, {}
            if not filtered_by_strike_ce.empty:
                if display_type == 'percentage':
                    daily_total_oi_ce = filtered_by_strike_ce.groupby(filtered_by_strike_ce['TIMESTAMP'].dt.date)['OPEN_INT'].sum()
                    filtered_by_strike_ce['OI_VALUE'] = filtered_by_strike_ce.apply(
                        lambda row: (row['OPEN_INT'] / daily_total_oi_ce[row['TIMESTAMP'].date()]) * 100 if daily_total_oi_ce[row['TIMESTAMP'].date()] != 0 else 0, axis=1
                    )
                else: # actual values
                    filtered_by_strike_ce['OI_VALUE'] = filtered_by_strike_ce['OPEN_INT'] / 1000 # Convert to thousands
                
                filtered_by_strike_ce['DATE_KEY'] = filtered_by_strike_ce['TIMESTAMP'].dt.strftime('%Y-%m-%d')
                pivot_table_ce = filtered_by_strike_ce.pivot_table(index='STRIKE_PRICE', columns='DATE_KEY', values='OI_VALUE', aggfunc='sum').fillna(0).round(2)
                pivot_table_ce = pivot_table_ce.reindex(columns=sorted(pivot_table_ce.columns))
                pivot_table_ce.columns = [pd.to_datetime(col).day for col in pivot_table_ce.columns]
                oi_data_ce = {"columns": pivot_table_ce.columns.tolist(), "index": pivot_table_ce.index.tolist(), "data": pivot_table_ce.to_dict(orient='records')}

                if display_type == 'percentage':
                    daily_total_volume_ce = filtered_by_strike_ce.groupby(filtered_by_strike_ce['TIMESTAMP'].dt.date)['TOT_TRADED_QTY'].sum()
                    filtered_by_strike_ce['VOLUME_VALUE'] = filtered_by_strike_ce.apply(
                        lambda row: (row['TOT_TRADED_QTY'] / daily_total_volume_ce[row['TIMESTAMP'].date()]) * 100 if daily_total_volume_ce[row['TIMESTAMP'].date()] != 0 else 0, axis=1
                    )
                else: # actual values
                    filtered_by_strike_ce['VOLUME_VALUE'] = filtered_by_strike_ce['TOT_TRADED_QTY'] / 1000 # Convert to thousands

                volume_pivot_table_ce = filtered_by_strike_ce.pivot_table(index='STRIKE_PRICE', columns='DATE_KEY', values='VOLUME_VALUE', aggfunc='sum').fillna(0).round(2)
                volume_pivot_table_ce = volume_pivot_table_ce.reindex(columns=sorted(volume_pivot_table_ce.columns))
                volume_pivot_table_ce.columns = [pd.to_datetime(col).day for col in volume_pivot_table_ce.columns]
                volume_data_ce = {"columns": volume_pivot_table_ce.columns.tolist(), "index": volume_pivot_table_ce.index.tolist(), "data": volume_pivot_table_ce.to_dict(orient='records')}

            # Process PE data
            oi_data_pe, volume_data_pe = {}, {}
            if not filtered_by_strike_pe.empty:
                if display_type == 'percentage':
                    daily_total_oi_pe = filtered_by_strike_pe.groupby(filtered_by_strike_pe['TIMESTAMP'].dt.date)['OPEN_INT'].sum()
                    filtered_by_strike_pe['OI_VALUE'] = filtered_by_strike_pe.apply(
                        lambda row: (row['OPEN_INT'] / daily_total_oi_pe[row['TIMESTAMP'].date()]) * 100 if daily_total_oi_pe[row['TIMESTAMP'].date()] != 0 else 0, axis=1
                    )
                else: # actual values
                    filtered_by_strike_pe['OI_VALUE'] = filtered_by_strike_pe['OPEN_INT'] / 1000 # Convert to thousands

                filtered_by_strike_pe['DATE_KEY'] = filtered_by_strike_pe['TIMESTAMP'].dt.strftime('%Y-%m-%d')
                pivot_table_pe = filtered_by_strike_pe.pivot_table(index='STRIKE_PRICE', columns='DATE_KEY', values='OI_VALUE', aggfunc='sum').fillna(0).round(2)
                pivot_table_pe = pivot_table_pe.reindex(columns=sorted(pivot_table_pe.columns))
                pivot_table_pe.columns = [pd.to_datetime(col).day for col in pivot_table_pe.columns]
                oi_data_pe = {"columns": pivot_table_pe.columns.tolist(), "index": pivot_table_pe.index.tolist(), "data": pivot_table_pe.to_dict(orient='records')}

                if display_type == 'percentage':
                    daily_total_volume_pe = filtered_by_strike_pe.groupby(filtered_by_strike_pe['TIMESTAMP'].dt.date)['TOT_TRADED_QTY'].sum()
                    filtered_by_strike_pe['VOLUME_VALUE'] = filtered_by_strike_pe.apply(
                        lambda row: (row['TOT_TRADED_QTY'] / daily_total_volume_pe[row['TIMESTAMP'].date()]) * 100 if daily_total_volume_pe[row['TIMESTAMP'].date()] != 0 else 0, axis=1
                    )
                else: # actual values
                    filtered_by_strike_pe['VOLUME_VALUE'] = filtered_by_strike_pe['TOT_TRADED_QTY'] / 1000 # Convert to thousands

                volume_pivot_table_pe = filtered_by_strike_pe.pivot_table(index='STRIKE_PRICE', columns='DATE_KEY', values='VOLUME_VALUE', aggfunc='sum').fillna(0).round(2)
                volume_pivot_table_pe = volume_pivot_table_pe.reindex(columns=sorted(volume_pivot_table_pe.columns))
                volume_pivot_table_pe.columns = [pd.to_datetime(col).day for col in volume_pivot_table_pe.columns]
                volume_data_pe = {"columns": volume_pivot_table_pe.columns.tolist(), "index": volume_pivot_table_pe.index.tolist(), "data": volume_pivot_table_pe.to_dict(orient='records')}

            all_stock_tables.append({
                "symbol": symbol,
                "expiry_day": expiry_day,
                "oi_data_ce": oi_data_ce,
                "volume_data_ce": volume_data_ce,
                "oi_data_pe": oi_data_pe,
                "volume_data_pe": volume_data_pe
            })

        if not all_stock_tables:
            return jsonify({"status": "success", "message": "No results found for the given strategy.", "data": []})
        
        return jsonify({"status": "success", "data": all_stock_tables})

    except Exception as e:
        import traceback
        print("ERROR: An unexpected error occurred in analyze_oi_shift_data:")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
