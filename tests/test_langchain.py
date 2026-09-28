"""LangGraph agent with a scripted chat model, recorded through the callback handler."""
import pytest

pytest.importorskip("langgraph")

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from runtape import Recorder, Trace


class ScriptedChat(BaseChatModel):
    replies: list

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kw: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.replies.pop(0)
        msg.usage_metadata = {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15}
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def get_balance(account: str) -> str:
    """Look up an account balance."""
    return f"{account}: $120.00"


@tool
def close_account(account: str) -> str:
    """Close an account."""
    raise PermissionError("needs manager approval")


def build(model):
    def agent(state: MessagesState):
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([get_balance, close_account], handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


def test_langgraph_run_is_recorded(tpath):
    model = ScriptedChat(
        replies=[
            AIMessage(content="", tool_calls=[{"id": "c1", "name": "get_balance", "args": {"account": "ACC-7"}}]),
            AIMessage(content="", tool_calls=[{"id": "c2", "name": "close_account", "args": {"account": "ACC-7"}}]),
            AIMessage(content="Your balance is $120. I couldn't close the account."),
        ]
    )
    rec = Recorder(tpath)
    out = build(model).invoke(
        {"messages": [SystemMessage("You are a bank bot."), HumanMessage("balance and close ACC-7")]},
        config={"callbacks": [rec.langchain()]},
    )
    rec.close()
    assert "couldn't close" in out["messages"][-1].content

    t = Trace.load(tpath)
    reqs = t.of_type("llm_request")
    assert len(reqs) == 3
    first = t.messages(reqs[0].id)
    assert first[0] == {"role": "system", "content": "You are a bank bot."}
    # later calls are delta-encoded on earlier ones
    assert reqs[1].payload.get("base") == reqs[0].id

    resp = t.of_type("llm_response")
    assert resp[0].payload["tool_calls"] == [{"id": "c1", "name": "get_balance", "arguments": {"account": "ACC-7"}}]
    assert resp[0].meta["tokens"] == {"input": 11, "output": 4}
    assert resp[-1].payload["text"].startswith("Your balance")

    calls = t.of_type("tool_call")
    assert [c.payload["name"] for c in calls] == ["get_balance", "close_account"]
    assert calls[0].payload["call_id"] == "c1"
    assert calls[0].meta["requested_by"] == resp[0].id
    results = t.of_type("tool_result")
    assert "$120.00" in str(results[0].payload["result"])
    # the failing tool shows up as an error result
    assert results[1].payload["error"]["type"] == "PermissionError"
    # and the last model call saw the tool output
    assert "$120.00" in str(t.context(resp[-1].id).messages)
    assert t.grep("manager approval")


def test_langchain_handler_never_raises(tpath):
    rec = Recorder(tpath)
    h = rec.langchain()
    rec.close()  # recorder closed: handler calls become no-ops, not crashes
    import uuid

    h.on_chat_model_start({}, [[HumanMessage("x")]], run_id=uuid.uuid4())
