# Falcon: Your AI-Powered Investment Assistant

Falcon is an AI-powered investment assistant built using LangGraph, LangChain, and Chainlit. It helps users make informed investment decisions by providing portfolio analysis, stock market trend analysis, and price checking.

## Features

* **Portfolio Retrieval:** Access and understand your current investment holdings.
* **Stock Analysis:** Analyze market trends and get insights into specific stocks.
* **Price Checking:** Quickly check the current price of stocks.
* **Personalized Recommendations:**  Receive tailored investment advice based on your risk tolerance and goals.  (Explain how personalization works, if applicable)
* **Interactive Chat Interface:**  A user-friendly chat interface powered by Chainlit.


## Architecture

Falcon uses a modular architecture combining several key technologies:

* **LangGraph:**  Orchestrates the workflow and decision-making process.
* **LangChain:** Provides tools and chains for interacting with external data sources and LLMs.
* **Chainlit:** Creates the interactive user interface.
* **Google Vertex AI:** Powers the underlying language model (Gemini).
* **Finnhub:** Provides real-time stock market data.
* **BigQuery:** Stores and retrieves portfolio information.

## Examples Queries
Interact with Falcon: Use the chat interface to ask questions about your portfolio, analyze stocks, and get investment recommendations.

* What stocks do I currently hold?
* What's the current price of AAPL?
* Am I making a profit or loss in my portfolio?
* Should I invest in TSLA?
* Should I sell off my Tesla stocks?

## Usage
**Run the Chainlit App:**
```bash
chainlit run falcon.py



