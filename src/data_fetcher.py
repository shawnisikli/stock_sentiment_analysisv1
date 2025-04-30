import yfinance as yf
import pandas as pd
from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

def load_api_keys():
    """Loads API keys from the .env file."""
    load_dotenv()
    news_api_key = os.getenv("NEWS_API_KEY")
    alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY") # Add ALPHA_VANTAGE_KEY=YOUR_KEY to .env if using
    if not news_api_key:
        print("Warning: NEWS_API_KEY not found in .env file.")
    # Add similar check for alpha_vantage_key if you plan to use it
    return news_api_key, alpha_vantage_key

def get_stock_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pandas.DataFrame: DataFrame containing historical stock data, or None if an error occurs.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}.")
            return None
        hist.reset_index(inplace=True) # Make Date a column
        hist['Date'] = pd.to_datetime(hist['Date']).dt.date # Keep only the date part
        return hist
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return None

def get_news_articles(query, from_date, to_date, language='en', sort_by='relevancy', page_size=100):
    """
    Fetches news articles related to a query within a date range using NewsAPI.

    Args:
        query (str): The search query (e.g., 'Apple stock').
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        language (str): Language of the articles (default: 'en').
        sort_by (str): Sorting criteria (default: 'relevancy'). Options: 'relevancy', 'popularity', 'publishedAt'.
        page_size (int): Number of results per page (max 100 for developer plan).

    Returns:
        list: A list of dictionaries, where each dictionary represents an article, or None if an error occurs.
              Returns an empty list if no articles are found.
    """
    print(f"Attempting to fetch news with query: '{query}'") # Added print
    print(f"Date range: {from_date} to {to_date}") # Added print
    news_api_key, _ = load_api_keys()
    if not news_api_key:
        print("Error: NewsAPI key not available. Cannot fetch news.") # Made error clearer
        return None

    try:
        newsapi = NewsApiClient(api_key=news_api_key)
        # NewsAPI free tier only allows searching articles up to one month old
        # Ensure from_date is not too far in the past if using free tier
        one_month_ago = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d') # Use 29 days to be safe
        print(f"One month ago date limit (approx): {one_month_ago}") # Added print
        if from_date < one_month_ago:
             print(f"Warning: NewsAPI free tier limits searches to the past month. Adjusting from_date from {from_date} to {one_month_ago}")
             from_date = one_month_ago

        print(f"Calling NewsAPI with: q='{query}', from='{from_date}', to='{to_date}', page_size={page_size}") # Added print
        all_articles = newsapi.get_everything(q=query,
                                              from_param=from_date,
                                              to=to_date,
                                              language=language,
                                              sort_by=sort_by,
                                              page_size=page_size) # Max 100 for free tier

        print(f"NewsAPI response status: {all_articles.get('status')}") # Added print
        if all_articles['status'] == 'ok':
            total_results = all_articles['totalResults']
            print(f"Found {total_results} articles for '{query}'")
            if total_results == 0:
                print("Warning: NewsAPI returned 0 articles for this query and date range.") # Added warning
            return all_articles['articles']
        else:
            error_code = all_articles.get('code')
            error_message = all_articles.get('message')
            print(f"Error fetching news from NewsAPI. Code: {error_code}, Message: {error_message}") # More detailed error
            return None
    except Exception as e:
        print(f"Exception occurred while connecting to NewsAPI: {e}") # Clarified exception source
        return None

# Placeholder for Alpha Vantage data fetching
def get_alpha_vantage_data(symbol):
    """Placeholder function to fetch data using Alpha Vantage."""
    _, alpha_vantage_key = load_api_keys()
    if not alpha_vantage_key:
        print("Alpha Vantage API key not found in .env file.")
        return None
    print(f"Fetching data for {symbol} using Alpha Vantage (implementation pending)...")
    # Add Alpha Vantage API call logic here
    return None

if __name__ == '__main__':
    # Example usage (for testing the module directly)
    ticker = 'AAPL'
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d') # Look back 30 days

    print(f"--- Testing Stock Data Fetching ({ticker}) ---")
    stock_data = get_stock_data(ticker, start_date, end_date)
    if stock_data is not None:
        print(f"Successfully fetched {len(stock_data)} rows of stock data.")
        print(stock_data.head())
    else:
        print("Failed to fetch stock data.")

    print(f"\n--- Testing News Article Fetching ({ticker}) ---")
    news_query = f"{ticker} stock"
    articles = get_news_articles(news_query, start_date, end_date)
    if articles is not None:
        print(f"Successfully fetched {len(articles)} articles.")
        if articles:
            print("First article title:", articles[0]['title'])
    else:
        print("Failed to fetch news articles.")

    # print("\n--- Testing Alpha Vantage (Placeholder) ---")
    # get_alpha_vantage_data(ticker)