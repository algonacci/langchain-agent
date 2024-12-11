import os
from dotenv import load_dotenv
import yfinance as yf
from langchain_groq import ChatGroq
import numpy as np
from typing import Dict, Any
import pandas as pd

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def fetch_stock_data(stock: str) -> yf.Ticker:
    """Fetch and cache stock data"""
    return yf.Ticker(stock)

def get_stock_analysis(stock: str) -> Dict[str, Any]:
    """Get comprehensive stock analysis"""
    try:
        # Fetch stock data
        stock_data = fetch_stock_data(stock)
        
        # Get basic info
        info = stock_data.info
        
        # Get recent history
        history = stock_data.history(period="1mo")
        
        # Get balance sheet
        balance_sheet = stock_data.balance_sheet.fillna(np.nan).replace({np.nan: None})
        
        # Prepare analysis text
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

4. Recent Performance:
- Latest Close: {history['Close'][-1] if not history.empty else 'N/A'}
- 1 Month Price Change: {((history['Close'][-1] / history['Close'][0] - 1) * 100).round(2) if not history.empty else 'N/A'}%
- Average Daily Volume: {history['Volume'].mean().round(0) if not history.empty else 'N/A'}

5. Balance Sheet Highlights (Most Recent):
- Total Assets: {balance_sheet.iloc[0].get('Total Assets', 'N/A')}
- Total Liabilities: {balance_sheet.iloc[0].get('Total Liabilities Net Minority Interest', 'N/A')}
- Total Equity: {balance_sheet.iloc[0].get('Total Equity Gross Minority Interest', 'N/A')}
- Cash and Equivalents: {balance_sheet.iloc[0].get('Cash And Cash Equivalents', 'N/A')}
"""
        
        return {"analysis": analysis}
    
    except Exception as e:
        return {"error": f"Error analyzing {stock}: {str(e)}"}

def analyze_stock(stock_symbol: str):
    """Analyze a single stock"""
    try:
        print(f"\nAnalyzing stock: {stock_symbol}")
        
        # Get analysis
        result = get_stock_analysis(stock_symbol)
        
        if "error" in result:
            print(result["error"])
        else:
            print(result["analysis"])
            
        # Optional: Use Groq for additional insights
        llm = ChatGroq(
            temperature=0,
            model_name="llama3-groq-70b-8192-tool-use-preview",
            groq_api_key=GROQ_API_KEY,
            max_tokens=1000
        )
        
        analysis_prompt = f"""
        Based on this stock analysis, provide 2-3 key insights:
        {result.get('analysis', '')}
        """
        
        try:
            insights = llm.invoke(analysis_prompt)
            print("\nAI Insights:")
            print(insights.content)
        except Exception as e:
            print(f"Note: AI insights unavailable: {e}")
            
    except Exception as e:
        print(f"Error analyzing {stock_symbol}: {str(e)}")

if __name__ == "__main__":
    stock = "BBCA.JK"
    analyze_stock(stock)