from langchain_core.tools import tool
from portfolio import query_portfolio
from google_grounding import google_ground
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain.schema import AIMessage
import os
import json
import re
import finnhub
from config import FINNHUB_API_KEY, LLM_MODEL



llm = ChatVertexAI(model_name=LLM_MODEL, temperature=0)


def get_stock_symbols(prompt: str) -> list:
    """Uses LLM to extract company names and convert them to ticker symbols."""

    prompt_template = """
    Extract the stock ticker symbols from the given text. If a company name is provided, convert it to its most common ticker symbol. If no symbols or company names are found, return an empty list in valid JSON format.

    Example 1:
    Text: "Get me the prices of Apple and Microsoft."
    Symbols: ["AAPL", "MSFT"]

    Example 2:
    Text: "What about Tesla, Google, and Amazon?"
    Symbols: ["TSLA", "GOOG", "AMZN"]

    Example 3:
    Text: "I want to know about IBM and Berkshire Hathaway."
    Symbols: ["IBM", "BRK-B"]

    Example 4:
    Text: "No stocks here."
    Symbols: []

    Return ONLY the valid JSON string representing the list of symbols. Do not include any other text or formatting like code blocks.

    Text: "{text}"
    Symbols:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    llm_output = llm.invoke(PROMPT.format(text=prompt))

    try:
        if isinstance(llm_output, AIMessage):
            llm_text = llm_output.content
        else:
            llm_text = str(llm_output)

        # More robust JSON extraction using regex
        match = re.search(r"\[.*\]", llm_text, re.DOTALL)  # Find JSON array using regex
        if match:
            json_string = match.group(0)
            symbols = json.loads(json_string)
            if isinstance(symbols, list):
                return [symbol.upper() for symbol in symbols]
            else:
                print(f"LLM did not return a list: {llm_text}")
                return []
        else:
            print(f"No JSON found in LLM output: {llm_text}")
            return []

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}. Full LLM Output: {llm_text}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

@tool
def price_checker(prompt: str) -> str:
    """Check the current price of one or more stocks using Finnhub. Accepts company names or ticker symbols."""
    print("Using Price Checker tool now")
    symbols = get_stock_symbols(prompt)

    if not symbols:
        print(f"No valid stock symbols or company names found in the input: {prompt}")
        return "No valid stock symbols or company names found in the input."

    results = []
    client = finnhub.Client(api_key=FINNHUB_API_KEY)
    for ticker_symbol in symbols:
        try:
            print(ticker_symbol)
            quote = client.quote(ticker_symbol)
            
            if not quote or quote['c'] is None:  # Check if quote and current price exist
                results.append(f"Could not retrieve price information for {ticker_symbol}. Check the ticker or Finnhub data.")
                continue

            current_price = quote['c']
            previous_close = quote.get('pc')  # Use .get to avoid KeyError if 'pc' is missing

            if previous_close is None:
                results.append(f"Could not retrieve previous close price for {ticker_symbol}.")
                continue

            change = current_price - previous_close
            percent_change = (change / previous_close) * 100 if previous_close != 0 else 0 # avoid division by zero

            results.append(
                f"{ticker_symbol}: Current Price: ${current_price:.2f}, Change: ${change:.2f} ({percent_change:.2f}%)"
            )

        except Exception as e:
            results.append(f"An error occurred for {ticker_symbol}: {e}")

    return "\n".join(results)

@tool
def portfolio_retriever(prompt: str) -> str:
    """Retrieves portfolio information. Information returned must be information on the portfolio. E.g. 100 units of TSLA stock, purchased at an avg price of $200"""
    print("Using Portfolio Retriever tool now")
    return query_portfolio(prompt)


@tool
def stock_analyser(prompt: str) -> str:
    """Analyzes stock market trends."""
    print("Using Stock Analyser tool now")
    return google_ground(prompt)


@tool
def normal_responder(qns: str) -> str:
    """Answer normal Question/Generic Question (e.g. Hi or who are you?)"""
    print("Using Normal Responder tool now")
    prompt_template=f"""You are Falcon, one of the most seasoned equity traders in the world.
        Your goal is to help to answer the user {qns} with comprehensive analysis based on what you have been trained on, or knowledge from Google Search or knowledge from internal proprietary investment research.
        You need to return a response that explains how you came up with that answer, backed by evidence that you used in coming up with the answer.
        The user is a day trader, risk tolerance is high. time horizon for trading is usually 1-2 months. Investment goal is to maximise the opportunity cost of the funds and reap maximum returns within the time horizon.
        """,
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    llm_output = llm.invoke(PROMPT.format(qns=qns))
    return(llm_output.content)
    
