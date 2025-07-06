# NSELib Flask App

This is a Flask web application designed to fetch, store, analyze, and visualize data from the National Stock Exchange (NSE) of India, primarily focusing on Futures & Options (F&O) data. It leverages the `nselib` Python library to interact with NSE's data.

## Features

*   **F&O Stock List Management:** Fetches and updates the list of F&O eligible stocks, storing them in a SQLite database.
*   **Custom Option Data Fetching:** Allows users to fetch historical option price and volume data for selected F&O symbols, instruments (e.g., EQUITY), option types (CE/PE), and date ranges.
*   **Data Storage:** Stores fetched F&O data in a SQLite database (`nselib_data.db`) for persistent storage and quick retrieval.
*   **Data Viewing:** Provides a web interface to view the stored option data with pagination.
*   **Data Cleanup:** Includes a utility to clean up stored data based on certain criteria (e.g., removing records with low traded quantity).
*   **Covered Call Analysis:** Implements a basic "Covered Call A+B" strategy analysis, identifying potential covered call opportunities based on interest and appreciation.
*   **Open Interest (OI) Shift Analysis:** Analyzes shifts in Open Interest and Volume for top strike prices of F&O stocks over time, providing insights into market sentiment.

## Project Structure

```
.
├── .DS_Store
├── app.py                  # Main Flask application logic
├── nselib_data.db          # SQLite database for storing NSE data
├── __pycache__/            # Python cache directory
│   └── app.cpython-312.pyc
└── templates/              # HTML templates for the web interface
    ├── analyze.html
    ├── base.html
    ├── covered_call.html
    ├── fetch_bulk_data.html
    ├── fetch_custom_data.html
    ├── fetch_data.html
    ├── index.html
    ├── insights.html
    ├── oi_shift.html
    ├── settings.html
    ├── view_data.html
    └── view_fno_stocks.html
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd flask_app
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize the database:**
    The `app.py` script will attempt to initialize the `nselib_data.db` database and create the necessary tables (`option_data` and `fno_stocks`) when the application starts.

## Usage

1.  **Run the Flask application:**
    ```bash
    python app.py
    ```

2.  **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/` (or the address shown in your terminal).

3.  **Fetch F&O Stocks:**
    Navigate to the appropriate section (e.g., `/fetch_data` and then trigger the F&O stock fetch) to populate the `fno_stocks` table. This is crucial for other features that rely on the list of active F&O symbols.

4.  **Fetch Custom Option Data:**
    Use the "Fetch Custom Data" section to specify symbols, instrument types, option types, and date ranges to download historical option data.

5.  **View Data:**
    The "View Data" section allows you to browse the stored option data.

6.  **Analysis:**
    Explore the "Analyze" section for "Covered Call" and "OI Shift" analyses.

## Dependencies

*   Flask
*   nselib
*   pandas
*   sqlite3 (built-in with Python)

## Contributing

Feel free to fork the repository, make improvements, and submit pull requests.

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]
