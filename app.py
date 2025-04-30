import gradio as gr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import io
import base64

# Add src directory to path to import modules
module_path = os.path.abspath(os.path.join('.'))
if (module_path not in sys.path):
    sys.path.append(module_path)

# Import functions from your src directory
try:
    from src.data_fetcher import get_stock_data, get_news_articles, load_api_keys
    from src.sentiment_analyzer import analyze_sentiment
except ImportError as e:
    # Handle error gracefully if run from a different directory or modules missing
    print(f"Error importing modules from src: {e}. Ensure app.py is in the project root and src/* exists.")
    # Define dummy functions if imports fail, so Gradio interface can still load
    def get_stock_data(*args, **kwargs): return None
    def get_news_articles(*args, **kwargs): return None
    def analyze_sentiment(*args, **kwargs): return None, None, None
    def load_api_keys(): return None, None


# --- Data Fetching and Processing Logic ---
# (Similar to the Streamlit version, but adapted for Gradio outputs)
# Added a comment below to trigger rebuild
def perform_analysis(ticker_symbol, start_date_str, end_date_str): # Renamed date inputs
    """Fetches data, analyzes sentiment, merges, and prepares outputs for Gradio."""
    if not ticker_symbol:
        return None, "Please enter a stock ticker.", None, None, None

    # Ensure API keys are loaded (needed for news)
    news_key, _ = load_api_keys()
    if not news_key:
         return None, "Error: NEWS_API_KEY not found in .env file. Cannot fetch news.", None, None, None

    # Convert Gradio date objects to strings
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    if start_date >= end_date:
        return None, "Error: Start date must be before end date.", None, None, None

    status_updates = f"Fetching data for {ticker_symbol} from {start_date_str} to {end_date_str}...\n"

    # 1. Fetch Stock Data
    stock_df = get_stock_data(ticker_symbol, start_date_str, end_date_str)
    if stock_df is None or stock_df.empty:
        status_updates += "Could not fetch stock data.\n"
        # Return early if essential data is missing
        return None, status_updates, None, None, None
    else:
        status_updates += f"Successfully fetched {len(stock_df)} days of stock data.\n"
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])


    # 2. Fetch News Articles
    articles_list = get_news_articles(ticker_symbol, start_date_str, end_date_str)
    if articles_list is None or not articles_list:
        status_updates += "Could not fetch news articles or none found.\n"
        news_df = pd.DataFrame()
    else:
        status_updates += f"Found {len(articles_list)} potential news articles.\n"
        news_df = pd.DataFrame(articles_list)
        if 'publishedAt' in news_df.columns:
            news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'])
            news_df['date'] = news_df['publishedAt'].dt.date
            news_df['date'] = pd.to_datetime(news_df['date']) # Convert date to datetime for merging
        else:
            status_updates += "Warning: News articles missing 'publishedAt' field.\n"
            news_df['date'] = None

    # 3. Sentiment Analysis (if news available)
    daily_sentiment = pd.DataFrame(columns=['date', 'avg_sentiment_score']) # Default empty
    if not news_df.empty and 'date' in news_df.columns and news_df['date'].notna().any():
        status_updates += f"Performing sentiment analysis on {len(news_df)} articles...\n"
        news_df['text_to_analyze'] = news_df['title'].fillna('') + ". " + news_df['description'].fillna('')
        # --- Apply sentiment analysis ---
        # This can be slow, consider progress updates if possible or running async
        sentiment_results = news_df['text_to_analyze'].apply(lambda x: analyze_sentiment(x) if pd.notna(x) else (None, None, None))
        news_df['sentiment_label'] = sentiment_results.apply(lambda x: x[0])
        news_df['sentiment_score'] = sentiment_results.apply(lambda x: x[1])
        status_updates += "Sentiment analysis complete.\n"

        # 4. Aggregate Sentiment
        valid_sentiment_df = news_df.dropna(subset=['sentiment_score', 'date'])
        if not valid_sentiment_df.empty:
             daily_sentiment = valid_sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
             daily_sentiment.rename(columns={'sentiment_score': 'avg_sentiment_score'}, inplace=True)
             status_updates += "Aggregated daily sentiment scores.\n"
        else:
            status_updates += "No valid sentiment scores found to aggregate.\n"

    # 5. Merge Data
    if not daily_sentiment.empty:
        merged_df = pd.merge(stock_df, daily_sentiment, left_on='Date', right_on='date', how='left')
        if 'date' in merged_df.columns:
            merged_df.drop(columns=['date'], inplace=True)
        status_updates += "Merged stock data with sentiment scores.\n"
    else:
        merged_df = stock_df.copy() # Keep stock data even if no sentiment
        merged_df['avg_sentiment_score'] = None # Add column with None
        status_updates += "No sentiment data to merge.\n"

    # 6. Calculate Price Change and Lagged Sentiment for Correlation
    merged_df['price_pct_change'] = merged_df['Close'].pct_change()
    merged_df['sentiment_lagged'] = merged_df['avg_sentiment_score'].shift(1)

    # --- Generate Outputs ---

    # Plot
    plot_object = None
    if not merged_df.empty:
        fig, ax1 = plt.subplots(figsize=(12, 6)) # Adjusted size for Gradio

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Stock Close Price', color=color)
        ax1.plot(merged_df['Date'], merged_df['Close'], color=color, label='Stock Price')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(axis='x', rotation=45)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10)) # Auto ticks

        if 'avg_sentiment_score' in merged_df.columns and merged_df['avg_sentiment_score'].notna().any():
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Average Sentiment Score', color=color)
            ax2.plot(merged_df['Date'], merged_df['avg_sentiment_score'], color=color, linestyle='--', marker='o', markersize=4, label='Avg Sentiment')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
            ax2.set_ylim(-1.1, 1.1) # Fix sentiment axis range

            # Combine legends
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
             ax1.legend(loc='upper left') # Only stock legend

        plt.title(f"{ticker_symbol} Stock Price vs. Average Daily News Sentiment")
        plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')
        fig.tight_layout()
        plot_object = fig # Return the figure object for Gradio plot component
        status_updates += "Generated plot.\n"


    # Correlation & Insights Text
    insights_text = "## Analysis Results\n\n"
    correlation = None
    if 'sentiment_lagged' in merged_df.columns and merged_df['sentiment_lagged'].notna().any() and merged_df['price_pct_change'].notna().any():
        correlation_df = merged_df[['sentiment_lagged', 'price_pct_change']].dropna()
        if not correlation_df.empty and len(correlation_df) > 1:
            correlation = correlation_df['sentiment_lagged'].corr(correlation_df['price_pct_change'])
            insights_text += f"**Correlation (Lagged Sentiment vs Price Change):** {correlation:.4f}\n"
            insights_text += "_Measures correlation between the previous day's average sentiment and the current day's price percentage change._\n\n"
        else:
            insights_text += "Correlation: Not enough overlapping data points to calculate.\n\n"
    else:
         insights_text += "Correlation: Sentiment or price change data missing.\n\n"

    # Simple Insights
    insights_text += "**Potential Insights (Not Financial Advice):**\n"
    if 'avg_sentiment_score' in merged_df.columns and merged_df['avg_sentiment_score'].notna().any():
        avg_sentiment_overall = merged_df['avg_sentiment_score'].mean()
        insights_text += f"- Average Sentiment (Overall Period): {avg_sentiment_overall:.3f}\n"

        if correlation is not None and pd.notna(correlation):
            if correlation > 0.15:
                insights_text += "- Positive correlation detected. Higher sentiment yesterday tended to correlate with price increases today.\n"
            elif correlation < -0.15:
                 insights_text += "- Negative correlation detected. Higher sentiment yesterday tended to correlate with price decreases today (or vice-versa).\n"
            else:
                insights_text += "- Weak correlation detected. Sentiment may not be a strong short-term driver for this period.\n"
    else:
        insights_text += "- No sentiment data available to generate insights.\n"

    insights_text += "\n**Disclaimer:** This analysis is automated and NOT financial advice. Many factors influence stock prices."
    status_updates += "Generated insights.\n"

    # Recent News DataFrame
    recent_news_df = pd.DataFrame()
    if not news_df.empty and 'publishedAt' in news_df.columns:
         # Select and format columns for display
         cols_to_show = ['publishedAt', 'title', 'sentiment_label', 'sentiment_score']
         # Ensure all columns exist before selecting
         cols_exist = [col for col in cols_to_show if col in news_df.columns]
         if cols_exist:
             recent_news_df = news_df.sort_values(by='publishedAt', ascending=False)[cols_exist].head(10)
             # Format date for display
             recent_news_df['publishedAt'] = recent_news_df['publishedAt'].dt.strftime('%Y-%m-%d %H:%M')
             status_updates += "Prepared recent news table.\n"


    return plot_object, insights_text, recent_news_df, status_updates, merged_df # Return merged_df for potential download


# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Stock Sentiment Analysis Dashboard")

    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(label="Stock Ticker", value="AAPL", placeholder="e.g., AAPL, GOOGL")
            # Use Textbox for dates, value should be string
            start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
            end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", value=datetime.now().strftime('%Y-%m-%d'))
            analyze_button = gr.Button("Analyze", variant="primary")
            status_output = gr.Textbox(label="Analysis Status", lines=5, interactive=False)
            # Optional: Add download button for the merged data
            download_data = gr.File(label="Download Merged Data (CSV)")

        with gr.Column(scale=3):
            plot_output = gr.Plot(label="Stock Price vs. Sentiment")
            insights_output = gr.Markdown(label="Analysis & Insights")
            news_output = gr.DataFrame(label="Recent News Headlines", headers=['Date', 'Title', 'Sentiment', 'Score'], wrap=True)

    # Hidden state to store the merged dataframe for download
    merged_df_state = gr.State(None)

    # Modify the wrapper function to accept strings and parse them
    def run_analysis_and_prepare_download(ticker, start_date_str, end_date_str):
        """Wrapper function to run analysis and prepare CSV for download."""
        try:
            # Validate and parse date strings here before passing to perform_analysis
            start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date_obj = datetime.strptime(end_date_str, '%Y-%m-%d').date()
        except ValueError:
             # Handle invalid date format input from textbox
             return None, "Error: Invalid date format. Please use YYYY-MM-DD.", None, "Error processing dates.", None, None

        # Pass the original strings to perform_analysis, as it expects strings now
        plot, insights, news, status, merged_df = perform_analysis(ticker, start_date_str, end_date_str)

        csv_path = None
        if merged_df is not None and not merged_df.empty:
            # Save to a temporary CSV file for Gradio download
            csv_path = "temp_merged_data.csv"
            merged_df.to_csv(csv_path, index=False)

        return plot, insights, news, status, merged_df, csv_path # Return path for download

    analyze_button.click(
        fn=run_analysis_and_prepare_download,
        inputs=[ticker_input, start_date_input, end_date_input], # Inputs are now textboxes
        outputs=[plot_output, insights_output, news_output, status_output, merged_df_state, download_data] # Update state and file output
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch() # App entry point