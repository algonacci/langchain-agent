import os
from dotenv import load_dotenv
import yfinance as yf
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
import numpy as np
from typing import Dict, Any
import pandas as pd

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Pre-fetch stock data once to optimize API calls
stock_cache = {}

def fetch_stock_data(stock: str) -> yf.Ticker:
    """Fetch and cache stock data"""
    if stock not in stock_cache:
        stock_cache[stock] = yf.Ticker(stock)
    return stock_cache[stock]

@tool
def get_stock_history(stock: str) -> Dict[str, Any]:
    """Get stock price history for the last month"""
    stock_data = fetch_stock_data(stock)
    hist = stock_data.history(period="1mo")
    return hist.to_dict()

@tool
def get_stock_info(stock: str) -> Dict[str, Any]:
    """Get general stock information and metrics"""
    stock_data = fetch_stock_data(stock)
    return stock_data.info

@tool
def get_stock_cashflow(stock: str) -> Dict[str, Any]:
    """Get stock cashflow statements"""
    stock_data = fetch_stock_data(stock)
    cashflow = stock_data.cashflow.fillna(np.nan).replace({np.nan: None})
    return cashflow.to_dict()

@tool
def get_balance_sheet(stock: str) -> Dict[str, Any]:
    """Get stock balance sheet data"""
    stock_data = fetch_stock_data(stock)
    balance_sheet = stock_data.balance_sheet.fillna(np.nan).replace({np.nan: None})
    return balance_sheet.to_dict()

def analyze_stock_data(stock: str) -> str:
    """Analyze stock data using the tools directly"""
    try:
        info = get_stock_info(stock)
        history = get_stock_history(stock)
        balance = get_balance_sheet(stock)
        
        # Convert history dict to DataFrame for easier processing
        hist_df = pd.DataFrame(history)
        
        # Get the most recent balance sheet data
        recent_balance = list(balance.values())[0] if balance else {}
        
        analysis = f"""
Stock Analysis for {stock}:

1. Basic Information:
- Company Name: {info.get('longName', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}

2. Current Market Data:
- Current Price: {info.get('currentPrice', 'N/A')} {info.get('currency', '')}
- 52 Week Range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}
- Market Cap: {info.get('marketCap', 'N/A')}
- Average Volume: {info.get('averageVolume', 'N/A')}

3. Key Metrics:
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Forward P/E: {info.get('forwardPE', 'N/A')}
- Price to Book: {info.get('priceToBook', 'N/A')}
- Dividend Yield: {info.get('dividendYield', 'N/A')}

4. Balance Sheet Highlights:
- Total Assets: {recent_balance.get('Total Assets', 'N/A')}
- Total Liabilities: {recent_balance.get('Total Liabilities Net Minority Interest', 'N/A')}
- Total Equity: {recent_balance.get('Total Equity Gross Minority Interest', 'N/A')}
- Cash and Equivalents: {recent_balance.get('Cash And Cash Equivalents', 'N/A')}
"""
        return analysis
        
    except Exception as e:
        return f"Error analyzing {stock}: {str(e)}"

def get_ai_insights(analysis: str) -> str:
    """Get AI insights using Groq"""
    try:
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-groq-70b-8192-tool-use-preview",
            groq_api_key=GROQ_API_KEY,
            max_tokens=1000
        )
        
        prompt = f"""
        Based on this stock analysis, provide 2-3 key insights:
        {analysis}
        Focus on the most important metrics and trends.
        """
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"AI insights unavailable: {e}"

def analyze_stocks(stock_symbols: list[str]):
    """Analyze multiple stocks"""
    for stock in stock_symbols:
        try:
            print(f"\nAnalyzing stock: {stock}")
            
            # Get basic analysis using tools
            analysis = analyze_stock_data(stock)
            print(analysis)
            
            # Get AI insights
            print("\nAI Insights:")
            insights = get_ai_insights(analysis)
            print(insights)
            
            print("\n" + "="*50 + "\n")
            
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue

if __name__ == "__main__":
    # Single stock analysis
    stock = "BBCA.JK"
    analyze_stocks([stock])
    
    # For multiple stocks, uncomment these lines:
    # stocks = ["BBCA.JK", "ADRO.JK", "GOTO.JK"]
    # analyze_stocks(stocks)