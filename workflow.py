import operator
import os
from typing import Sequence

from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated
from tools.hybird_search import search_document_knowledge

load_dotenv()

ROUTER_URI_LLM = os.getenv("ROUTER_URI_LLM")
ROUTER_KEY = os.getenv("ROUTER_KEY")
DB_URI = os.getenv("DB_URI")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    summary: str
    document_id: int

tools = [search_document_knowledge]

model = ChatOpenAI(
    base_url=ROUTER_URI_LLM,
    api_key=ROUTER_KEY,
    model="gemini-2.5-flash",
    temperature=0
)

llm = model.bind_tools(tools)

def call_model(state: State):
    messages = list(state["messages"])
    doc_id = state.get("document_id")
    summary = state.get("summary", "")

    if summary:
        system_summary = SystemMessage(
            content=f"Đây là tóm tắt của các tin nhắn cũ trong quá khứ để bạn nắm ngữ cảnh: {summary}")
        messages = [system_summary] + messages

    if doc_id:
        system_doc = SystemMessage(
            content=f"Bạn đang hỗ trợ người dùng tra cứu tài liệu có ID là {doc_id}. Hãy dùng đúng ID này khi gọi công cụ.")
        messages = [system_doc] + messages

    response = llm.invoke(messages)

    return {"messages": [response]}

def should_summarize(state: State):
    messages = state["messages"]
    if len(messages) > 10:
        return "summarize"
    return "agent"

def summarize(state: State):
    messages = state["messages"]
    existing_summary = state.get("summary", "")
    summary_prompt = (
        f"Hãy tạo một bản tóm tắt ngắn gọn cho cuộc trò chuyện dưới đây. "
        f"Kết hợp cả nội dung tóm tắt cũ nếu có.\n\n"
        f"Tóm tắt cũ: {existing_summary}\n\n"
        f"Tin nhắn mới:\n"
    )
    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in messages])
    response = model.invoke(summary_prompt + conversation_text)
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2] if m.id is not None]
    return {
        "summary": response.content,
        "messages": delete_messages
    }

def should_continue(state: State):
    last_messages = state["messages"][-1]
    if last_messages.tool_calls:
        return "tools"
    return END

workflow = StateGraph(State)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("summary", summarize)

workflow.add_conditional_edges(
    START,
    should_summarize,
    {
        "summary": "summary",
        "agent": "agent"
    }
)

workflow.add_edge("summary", "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge("tools", "agent")

checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = checkpointer_cm.__enter__()
checkpointer.setup()

graph = workflow.compile(checkpointer=checkpointer)