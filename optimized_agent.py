import os
from dotenv import load_dotenv
import yfinance as yf
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
import numpy as np
from typing import Dict, Any
from langchain_core.messages import SystemMessage

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

def create_stock_analysis_agent():
    """Create and configure the stock analysis agent"""
    # Define tools
    tools = [
        get_stock_history,
        get_stock_info,
        get_stock_cashflow,
        get_balance_sheet,
    ]

    # Enhanced system prompt
    system_message = """You are an intelligent financial advisor specializing in stock analysis. 
    For each stock, analyze:
    1. Recent price trends and trading volumes
    2. Key financial metrics and ratios
    3. Cash flow health
    4. Balance sheet strength
    
    Provide clear, concise insights that would be valuable for investment decisions.
    """

    # Define the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    # Configure the LLM with more specific parameters
    llm = ChatGroq(
        temperature=0,
        model_name="llama3-groq-70b-8192-tool-use-preview",
        groq_api_key=GROQ_API_KEY,
        max_tokens=4096,  # Add explicit token limit
        streaming=True,    # Enable streaming for better stability
    )

    # Create the agent with error handling
    try:
        agent = create_tool_calling_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True  # Add error handling
        )
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None

def analyze_stocks(stock_symbols: list[str]):
    """Analyze multiple stocks with error handling"""
    agent_executor = create_stock_analysis_agent()
    if not agent_executor:
        print("Failed to create agent executor")
        return

    for stock in stock_symbols:
        try:
            print(f"\nAnalyzing stock: {stock}")
            # Pre-fetch the stock data
            fetch_stock_data(stock)
            
            # Run analysis with timeout
            result = agent_executor.invoke(
                {"input": f"Analyze the financial performance and metrics for {stock}"},
                config={"timeout": 300}  # 5-minute timeout
            )
            
            print(f"\nAnalysis for {stock}:")
            print(result['output'] if 'output' in result else result)
            
        except Exception as e:
            print(f"Error analyzing {stock}: {e}")
            continue

if __name__ == "__main__":
    # List of stocks to analyze
    stocks = [
        "BBCA.JK",
        # "ADRO.JK",
        # "GOTO.JK"
    ]
    
    analyze_stocks(stocks)