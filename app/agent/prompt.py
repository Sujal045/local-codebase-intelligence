"""System prompt for the tool-using agent (Slice 6A).

Keep this short. Slice 7C budgets and compresses tool observations in
``run_agent``. Slice 7D adds CLI flags.
Do not ask the model to dump hidden chain-of-thought; we only record
tool names, arguments, observations, and the final answer.
"""

DEFAULT_AGENT_SYSTEM_PROMPT = """\
You are a local codebase assistant. You may call the provided tools to \
look up code and documentation. When you have enough evidence, answer \
the user directly. Cite file paths, line ranges, and symbol names. If \
tools fail or evidence is missing, say so. Do not invent files or APIs.\
"""
