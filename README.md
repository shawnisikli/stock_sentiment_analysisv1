---
title: Stock Sentiment Analysis
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.29.0 # Or check your installed version with pip show gradio
python_version: 3.10 # Adjust if necessary
app_file: app.py
license: apache-2.0 # Or choose another license if preferred
---

# Stock Sentiment Analysis Dashboard

This application fetches stock price data and related news articles, performs sentiment analysis on the news using FinBERT, and visualizes the relationship between stock price and news sentiment.

**Features:**
*   Enter a stock ticker and date range.
*   Visualizes stock price vs. average daily sentiment.
*   Calculates correlation between lagged sentiment and price changes.
*   Displays recent news headlines with their sentiment scores.

**Disclaimer:** This is a demo application and not financial advice.