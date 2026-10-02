"""
LangGraph + LangChain agent that answers questions about the Titanic dataset
using pandas tools.

The model is configured with the MODEL env var (default: Gemini 3.5 Flash).
langchain-google-genai reads the API key from the GEMINI_API_KEY env var.
To use another provider, install its package (e.g. langchain-openai) and change MODEL:
    MODEL="openai:gpt-4o-mini"
"""

import os

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------
DATASET_URL = os.getenv(
    "DATASET_URL", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
df = pd.read_csv(DATASET_URL)


# --------------------------------------------------------------------------
# 2. Tools (LangChain): functions the model can decide to call
# --------------------------------------------------------------------------
@tool
def schema() -> str:
    """Returns the dataset's columns, their types, null counts and 3 sample rows."""
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum()})
    return f"Rows: {len(df)}\n\n{info.to_string()}\n\nSample:\n{df.head(3).to_string()}"


@tool
def statistics(column: str) -> str:
    """Descriptive statistics for a column (mean, min, max, or frequencies if categorical)."""
    if column not in df.columns:
        return f"Column '{column}' does not exist. Columns: {list(df.columns)}"
    series = df[column]
    if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
        return series.describe().to_string()
    return series.value_counts(dropna=False).to_string()


@tool
def survival_rate(group_by: list[str]) -> str:
    """Survival rate (0 to 1) and passenger count, grouped by one or more columns.
    Example: group_by=["sex", "class"]."""
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        return f"Nonexistent columns: {missing}. Columns: {list(df.columns)}"
    res = df.groupby(group_by, observed=True)["survived"].agg(rate="mean", passengers="count")
    return res.round(3).to_string()


@tool
def filter_passengers(condition: str) -> str:
    """Filters passengers with a pandas.query expression and returns how many match,
    their survival rate and up to 5 rows. Example: "age < 18 and pclass == 3"."""
    try:
        sub = df.query(condition)
    except Exception as e:  # the error goes back to the model so it can fix the expression
        return f"Error in condition: {e}"
    if sub.empty:
        return "No passengers match the condition."
    return (
        f"Passengers: {len(sub)}\n"
        f"Survival rate: {sub['survived'].mean():.3f}\n\n{sub.head(5).to_string()}"
    )


TOOLS = [schema, statistics, survival_rate, filter_passengers]

INSTRUCTIONS = SystemMessage(
    "You are a data analyst. You answer questions about the Titanic dataset "
    "using the available tools; never make up figures. If you don't know the "
    "columns, check the schema first. Keep answers brief."
)


# --------------------------------------------------------------------------
# 3. Graph (LangGraph)
#
#    START -> agent --(did it request tools?)--> tools -> agent ...
#                   \--(no)--> END
# --------------------------------------------------------------------------
def build_graph(model):
    model_with_tools = model.bind_tools(TOOLS)

    def agent(state: MessagesState):
        response = model_with_tools.invoke([INSTRUCTIONS] + state["messages"])
        return {"messages": [response]}  # appended to the state's history

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" if the model requested any; otherwise to END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    # No checkpointer: each API request is an independent, stateless conversation
    return graph.compile()


model = init_chat_model(os.getenv("MODEL", "google_genai:gemini-3.5-flash"), temperature=0)
app = build_graph(model)


# --------------------------------------------------------------------------
# 4. Entry point used by the API
# --------------------------------------------------------------------------
def ask(question: str) -> dict:
    """Runs the agent on a single question and returns the answer plus the tools it used."""
    state = app.invoke({"messages": [HumanMessage(question)]}, {"recursion_limit": 25})
    tool_calls = [
        {"name": tc["name"], "args": tc["args"]}
        for msg in state["messages"]
        for tc in getattr(msg, "tool_calls", None) or []
    ]
    return {"answer": state["messages"][-1].text, "tool_calls": tool_calls}
