import chainlit as cl
import operator
from langchain_google_vertexai import ChatVertexAI
from typing import Annotated, List, Tuple, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from tools import portfolio_retriever, stock_analyser, normal_responder, price_checker



# --- Configuration and Constants ---
LLM_MODEL = 'gemini-2.0-flash-exp'
AGENT_PROMPT = """You are Falcon, a professional expert investment trader who can make investment recommendations and analysis. You are NOT an LLM or AI chatbot.
Help to execute the task that you are assigned to, and return the response in a clear and concise manner.
The user is a day trader, risk tolerance is high. time horizon for trading is usually 1-2 months. Investment goal is to maximise the opportunity cost of the funds and reap maximum returns within the time horizon.
**IMPORTANT** If you are asked for guidance/advice or need to make a recommendation, you MUST provide comprehensive advice or make a recommendation based on the info that you have gathered"""


# --- LLM and Tools ---
llm = ChatVertexAI(model_name=LLM_MODEL, temperature=0)
tools = [portfolio_retriever, stock_analyser, normal_responder, price_checker]
agent_executor = create_react_agent(llm, tools, state_modifier=AGENT_PROMPT)

# --- Data Models ---
class Plan(BaseModel):
    """Plan to follow."""
    steps: List[str] = Field(description="Steps to follow, in order.")

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple[str, str]], operator.add]
    response: Optional[str]
    intermediate_responses: List[str]

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform (for replanning)."""
    response: Optional[Response] = None
    plan: Optional[Plan] = None


# --- Prompts ---
PLANNER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert in deciphering questions and creating step-by-step plans.
Based on the given objective, create a simple plan. Each step should be a distinct task that, when executed, will lead to the correct answer. Avoid superfluous steps.
Use these guidelines to choose the right tool:
- Portfolio retrieval (e.g., "What's my portfolio?", "What are my holdings?", "Last trade on Nvidia?"): Use the {{portfolio_retriever}} tool.
- Check the current price of a stock using the stock symbol(e.g. current price of GOOG): Use the {{price_checker}} tool. If there are multiple stocks to check, break down into multiple steps to do multiple function calls to check current price for individual stock.
- Equity/market analysis (e.g., "Will Nvidia rise?", "Current stock price?", "Is Intel a buy?", "What are the risks?"): Use the {{stock_analyser}} tool.
- General/non-financial questions (e.g., "Hi", "Who are you?"): Use the {{normal_responder}} tool.
The final step's result should be the final answer. Ensure each step has enough information; do not skip steps.

Example Qns: Should I sell off my Nvidia stocks now?
Example Plan:
    Step 1: Check the number of Nvidia stocks and average purchase price in portfolio using {{portfolio_retriever}} tool
    Step 2: Check the current price of Nvidia stock using {{price_checker}} tool
    Step 3: Analyse how Nvidia stock is doing in today's market and is it recommended to sell or hold using the {{stock_analyser}} tool
    Step 4: Based on current price of Nvidia stock and purchase price, calculate much will user lose or profit if selling today? 
    Step 5: Combine all pieces of information from prior steps to put up a recommendation
""",
        ),
        ("placeholder", "{messages}"),
    ]
)

REPLANNER_PROMPT = ChatPromptTemplate.from_template(
    """Create a step-by-step plan for the given objective.
Each step should be a distinct task. The final step's result should be the final answer. Do not skip steps.

Your objective:
{input}

Your original plan:
{plan}

Completed steps:
{past_steps}

Update the plan. Include only the steps that still NEED to be done, incorporating data from previous steps. Do NOT include previously completed steps."""
)

# --- Chains and Agents ---
planner = PLANNER_PROMPT | llm.with_structured_output(Plan)
replanner = REPLANNER_PROMPT | llm.with_structured_output(Act)

# --- Mis ---
def clean_newlines(text: str) -> str:
    """Removes extra newline characters from a string."""
    return text.replace("\n\n", "\n")

# --- Workflow Nodes ---
async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    with cl.Step(name="Generated Plan"):
        await cl.Message(content="**Generated Plan:**").send()
        for i, step in enumerate(plan.steps):
            await cl.Message(content=f" {step}").send()
    return {"plan": plan.steps, "intermediate_responses": []}

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    if not plan:
        return {"response": "No more steps in the plan."}

    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    step_number = len(state.get("past_steps", [])) + 1
    task_formatted = f"""For the following plan: {plan_str}\n\nYou are tasked with executing step {step_number}, {task}."""

    with cl.Step(name=f"Executing: {task}"):
        await cl.Message(content=f"**Executing** {task}").send()
        agent_response = await agent_executor.ainvoke({"messages": [("user", task_formatted)]})
        final_response = agent_response["messages"][-1].content

        await cl.Message(content=final_response).send()

    return {
        "past_steps": state.get("past_steps", []) + [(task, final_response)],
        "plan": plan[1:],
        "intermediate_responses": state.get("intermediate_responses", []) + [final_response]
    }

async def replan_step(state: PlanExecute):
    all_responses = "\n".join(state["intermediate_responses"])
    all_steps = "\n".join([f"{step}: {response}" for step, response in state["past_steps"]])
    context = f"Here is the information gathered from the previous steps:\n{all_steps}\n\nHere are the direct responses from the tools:\n{all_responses}"

    output = await replanner.ainvoke({**state, "input": context})
    if output.response:
        cleaned_response = clean_newlines(output.response.response)
        with cl.Step(name="Final Response"):
            await cl.Message(content="**Final Response:**").send()
            await cl.Message(content=cleaned_response).send()
        return {"response": cleaned_response}
    else:
        return {"plan": output.plan.steps}

def should_end(state: PlanExecute):
    return END if "response" in state and state["response"] is not None else "agent"

# --- Workflow Definition ---
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges("replan", should_end, {"agent": "agent", END: END})
app = workflow.compile()

# --- Chainlit Interface ---
@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):
    
    config = {"recursion_limit": 50}
    async for event in app.astream(
        {"input": message.content, "plan": [], "past_steps": [], "response": None, "intermediate_responses": []},
        config=config,
    ):
        if "response" in event:
            # Final response is handled in replan_step, no action needed here
            pass
        elif "plan" in event:
            # Plan generation messages are handled in plan_step, no action needed here
            pass
        elif "past_steps" in event:
            # Execution step messages are handled in execute_step, no action needed here
            pass