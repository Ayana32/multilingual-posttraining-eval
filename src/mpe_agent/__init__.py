"""v1 agent-injection evaluation package (docs/v1-spec.md).

Ports the Phase 3/3b agent harness (phase3/harness.py) to run against
OpenAI-compatible endpoints. Phase 3 code is reused, never modified: the
outcome rules, the 3-turn loop shape, the fixed mock tool results and the
record schema are the same, with only the tool-call *source* changed from
<tool_call> text parsing to structured `tool_calls` (Section 4).
"""
