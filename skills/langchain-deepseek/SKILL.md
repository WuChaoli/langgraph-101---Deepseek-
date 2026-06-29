---
name: langchain-deepseek
description: Use when working on this LangGraph 101 project with DeepSeek-backed LangChain/LangGraph code, especially changes involving utils/models.py, langchain-deepseek, model.bind_tools, model.with_structured_output, tool_choice errors, json_mode parsing, or DeepSeek thinking-mode compatibility.
---

# LangChain DeepSeek

## Core Rule

Treat DeepSeek thinking-mode compatibility as a first-class constraint. Before changing LangChain or LangGraph model calls, search the touched scope for:

```bash
rg -n "with_structured_output|bind_tools|tool_choice|deepseek" <paths>
```

## Known Failure Pattern

This project currently uses DeepSeek in [utils/models.py](/Users/wuchaoli/Desktop/codespace/langgraph-101/utils/models.py):

```python
model = init_chat_model("deepseek:deepseek-v4-flash")
```

DeepSeek thinking mode can reject LangChain calls that send `tool_choice`:

```text
Thinking mode does not support this tool_choice
```

Common triggers:

```python
model.bind_tools(tools, tool_choice="any")
model.with_structured_output(SomeSchema)
```

## Tool Calling

Do not force tool selection unless a tested DeepSeek configuration proves it works.

Prefer:

```python
llm_with_tools = model.bind_tools(tools)
```

If the code needs serial tool calls:

```python
llm_with_tools = model.bind_tools(tools, parallel_tool_calls=False)
```

After removing forced `tool_choice`, handle the case where the model returns a normal message with no tool calls:

```python
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        ...
    return END
```

Without this default branch, LangGraph conditional routing may fail with `KeyError: None`.

## Structured Output

For DeepSeek-backed structured output, prefer JSON mode:

```python
llm_router = model.with_structured_output(RouterSchema, method="json_mode")
```

Always add explicit JSON instructions to the prompt. Include every required Pydantic field:

```text
请只返回 JSON，格式为 {"reasoning": "分类理由", "classification": "ignore|respond|notify"}。
```

If the instruction is inside an f-string, escape braces:

```python
f"""请只返回 JSON，格式为 {{"reasoning": "分类理由", "classification": "ignore|respond|notify"}}。"""
```

Apply the same check to every schema, not only routers. Examples include memory/profile schemas such as `UserProfile`.

## Verification

Use the smallest reproducible path first. For notebooks, execute only the required cells in order with `.venv/bin/python` before running the whole notebook flow.

Minimum checks after a DeepSeek compatibility edit:

```bash
rg -n "with_structured_output|bind_tools|tool_choice" agents notebooks utils
.venv/bin/python - <<'PY'
from agents.email_agent.graph import graph
print(type(graph).__name__)
PY
```

When a notebook was the failure surface, reproduce the exact failing cell sequence and confirm the original exception is gone before claiming success.
