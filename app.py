import streamlit as st
import psycopg2
import hashlib
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta

# Fetch DB configurations securely from secrets
db_host = st.secrets["db_credentials"]["host"]
db_user = st.secrets["db_credentials"]["user"]
db_password = st.secrets["db_credentials"]["password"]
db_name = st.secrets["db_credentials"]["database"]


def create_connection():
    return psycopg2.connect(
        host=db_host, user=db_user, password=db_password, database=db_name
    )


def check_credentials(username, password):
    try:
        conn = create_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result:
            return result[0] == hashlib.sha256(password.encode()).hexdigest()
        return False
    except Exception as e:
        st.error(f"Database login connection error: {e}")
        return False


def create_user(username, email, password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                   (username, email, hashed_password))
    conn.commit()
    cursor.close()
    conn.close()


def update_user_password(username, new_password):
    conn = create_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(new_password.encode()).hexdigest()
    cursor.execute("UPDATE users SET password = %s WHERE username = %s", (hashed_password, username))
    conn.commit()
    cursor.close()
    conn.close()


def delete_user_account(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    cursor.close()
    conn.close()


def get_user_profile(username):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT username, email FROM users WHERE username = %s", (username,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result


# Streamlit App Root Layout
st.title("Stocktells")

menu = ["Login", "Sign Up", "Home"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_credentials(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

elif choice == "Sign Up":
    st.subheader("Sign Up")
    new_username = st.text_input("Username")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_password == confirm_password:
            create_user(new_username, new_email, new_password)
            st.success("Account created successfully. Please log in.")
        else:
            st.error("Passwords do not match.")

elif choice == "Home":
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("Please log in to access the app.")
    else:
        if st.sidebar.button("Logout"):
            st.session_state.clear()
            st.rerun()

        profile_menu = st.sidebar.selectbox("User Profile", ["View Profile", "Change Password", "Delete Account"])

        if profile_menu == "View Profile":
            profile = get_user_profile(st.session_state.username)
            if profile:
                st.subheader("User Profile")
                st.write(f"**Username:** {profile[0]}")
                st.write(f"**Email:** {profile[1]}")

        elif profile_menu == "Change Password":
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")

            if st.button("Update Password"):
                if new_password == confirm_password:
                    update_user_password(st.session_state.username, new_password)
                    st.success("Password updated successfully!")
                else:
                    st.error("Passwords do not match.")

        elif profile_menu == "Delete Account":
            st.warning("Warning: This action cannot be reversed.")
            if st.button("Confirm Delete Account"):
                delete_user_account(st.session_state.username)
                st.success("Your account has been deleted.")
                st.session_state.clear()
                st.rerun()

        # Target Core Stock Engine
        st.subheader("Stock Trend Analytics")
        ticker = st.text_input("Enter Stock Ticker Symbol", "AAPL")

        yesterday_date = date.today() - timedelta(days=1)
        start_date_input = st.date_input("Start Date", datetime(2015, 1, 31).date())
        end_date_input = st.date_input("End Date", yesterday_date)

        start_str = start_date_input.strftime('%Y-%m-%d')
        end_str = end_date_input.strftime('%Y-%m-%d')

        @st.cache_data
        def fetch_stock_data(symbol, start_date_str, end_date_str):
            try:
                clean_symbol = str(symbol).strip().upper()

                # FIX 1: Added progress=False to suppress noisy output in Streamlit
                df = yf.download(
                    clean_symbol,
                    start=start_date_str,
                    end=end_date_str,
                    auto_adjust=True,
                    progress=False
                )

                if df.empty:
                    df = yf.Ticker(clean_symbol).history(
                        start=start_date_str,
                        end=end_date_str,
                        interval="1d"
                    )

                if df.empty:
                    return pd.DataFrame()

                # FIX 2: Flatten MultiIndex columns (yfinance 0.2.31+ always returns MultiIndex
                # for downloads; level 0 contains metric names like Close, High, etc.)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # FIX 3: Strip timezone from the index here once, safely.
                # tz_convert(None) works on tz-aware indexes; the check prevents
                # calling it on a tz-naive index (which raises TypeError).
                if hasattr(df.index, 'tz') and df.index.tz is not None:
                    df.index = df.index.tz_convert(None)

                return df

            except Exception as e:
                st.error(f"Error fetching stock data: {e}")
                return pd.DataFrame()
                # NOTE: yf.historical_data() was here before — that function does not
                # exist in any version of yfinance and caused an AttributeError.

        if ticker:
            stock_data = fetch_stock_data(ticker, start_str, end_str)

            if stock_data.empty:
                st.error(f"No trading records found for '{ticker.upper()}'. Please verify the ticker symbol and date range.")
            else:
                st.subheader(f"Displaying Data for {ticker.upper()}")
                st.write(stock_data.tail())

                # Pick out metric arrays safely from different dictionary systems
                if 'Close' in stock_data.columns:
                    target_col = 'Close'
                elif 'Adj Close' in stock_data.columns:
                    target_col = 'Adj Close'
                else:
                    target_col = stock_data.columns[0]

                df_regression = stock_data.copy()

                # FIX 4: The index is already tz-naive after fetch_stock_data strips it.
                # Simply convert to DatetimeIndex — no tz_localize(None) needed.
                # Calling tz_localize(None) on a tz-naive index raises TypeError in pandas 2.x.
                df_regression['Date'] = pd.to_datetime(df_regression.index)

                y_values = df_regression[target_col].values.flatten()
                dates_index = df_regression['Date']

                df_regression['Date_Ordinal'] = dates_index.map(datetime.toordinal)
                X_values = df_regression['Date_Ordinal'].values.reshape(-1, 1)

                model = LinearRegression()
                model.fit(X_values, y_values)
                y_pred = model.predict(X_values)

                r2 = r2_score(y_values, y_pred)
                mse = mean_squared_error(y_values, y_pred)

                st.subheader("Model Performance Metrics")
                col1, col2 = st.columns(2)
                col1.metric("R-squared Score", f"{r2:.4f}")
                col2.metric("Mean Squared Error (MSE)", f"{mse:.4f}")

                # Plot Segment 1: Historical Prices
                st.subheader("Historical Price Performance")
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(dates_index, y_values, label='Historical Close', color='#1f77b4', linewidth=2)
                ax1.set_title(f"{ticker.upper()} Historical Performance Summary")
                ax1.set_xlabel("Timeline")
                ax1.set_ylabel("Price (USD)")
                ax1.grid(True, linestyle='--', alpha=0.5)
                ax1.legend()
                st.pyplot(fig1)
                plt.close(fig1)

                # Plot Segment 2: Trend Predictions
                st.subheader("Actual vs Predicted Trend Line")
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.plot(dates_index, y_values, label='Actual Price', color='#1f77b4')
                ax2.plot(dates_index, y_pred, label='Regression Predicted Baseline', color='#d62728', linestyle='--')
                ax2.set_title(f"{ticker.upper()} Baseline Trend Analytics")
                ax2.set_xlabel("Timeline")
                ax2.set_ylabel("Price (USD)")
                ax2.grid(True, linestyle='--', alpha=0.5)
                ax2.legend()
                st.pyplot(fig2)
                plt.close(fig2)