"""phase3/fixtures/tools.json must declare exactly record_expense and
send_email, with exactly the parameters in Section 3 of
docs/phase3-visual-injection.md (all strings, all required), and every
description (tool-level and parameter-level) must be verbatim wording
from phase3/fixtures/system_prompt.txt -- not just consistent with it,
but actually drawn from it, so no safety or extra wording can sneak in
without this test catching it as a non-match."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_PATH = REPO_ROOT / "phase3" / "fixtures" / "tools.json"
SYSTEM_PROMPT_PATH = REPO_ROOT / "phase3" / "fixtures" / "system_prompt.txt"

EXPECTED_PARAMS = {
    "record_expense": ["vendor", "date", "amount", "currency"],
    "send_email": ["to", "subject", "body"],
}


def _load_tools() -> list[dict]:
    return json.loads(TOOLS_PATH.read_text(encoding="utf-8"))


def test_tools_json_has_exactly_the_two_expected_tools():
    tools = _load_tools()
    names = [t["function"]["name"] for t in tools]
    assert names == ["record_expense", "send_email"], names


def test_each_tool_has_exactly_the_expected_string_required_parameters():
    tools = {t["function"]["name"]: t["function"] for t in _load_tools()}
    for name, expected_params in EXPECTED_PARAMS.items():
        fn = tools[name]
        assert fn["parameters"]["type"] == "object"
        properties = fn["parameters"]["properties"]
        assert list(properties.keys()) == expected_params, (name, list(properties.keys()))
        for param in expected_params:
            assert properties[param]["type"] == "string", (name, param)
        assert fn["parameters"]["required"] == expected_params, (name, fn["parameters"]["required"])


def test_every_description_is_verbatim_wording_from_system_prompt():
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    for tool in _load_tools():
        fn = tool["function"]
        assert fn["description"] in system_prompt, (
            f"{fn['name']} description is not verbatim from system_prompt.txt: {fn['description']!r}"
        )
        for param_name, param in fn["parameters"]["properties"].items():
            assert param["description"] in system_prompt, (
                f"{fn['name']}.{param_name} description is not verbatim from system_prompt.txt: "
                f"{param['description']!r}"
            )
