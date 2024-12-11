import os
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@tool
def get_stock_history(stock):
    """
    Get stock history
    """
    stock_data = yf.Ticker(stock)
    hist = stock_data.history(period="1mo")
    return hist

@tool
def get_stock_info(stock):
    """
    Get stock info
    """
    stock_data = yf.Ticker(stock)
    return stock_data.info

@tool
def get_stock_cashflow(stock):
    """
    Get stock cashflow
    """
    stock_data = yf.Ticker(stock)
    return stock_data.cashflow
    
@tool
def get_balance_sheet(stock):
    """
    Get balance sheet
    """
    stock_data = yf.Ticker(stock)
    return stock_data.balance_sheet

tools = [
    get_stock_history,
    get_stock_info,
    get_stock_cashflow,
    get_balance_sheet,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an intelligent financial advisor.
            """
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)



llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


stocks = [
    "BBCA.JK",
    "ADRO.JK",
    "GOTO.JK"
]

for stock in stocks:
    print("Stock: ", stock)
    result = agent_executor.invoke({"input": stock})