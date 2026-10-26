"""The agent must degrade gracefully: bad LLM output, unknown tools and a missing
Ollama must never crash the dashboard."""
import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

pytest.importorskip("mcp")
from agent.discovery_mcp_agent import DiscoveryMCPAgent, LLMUnavailableError  # noqa: E402

DEAD_OLLAMA = "http://127.0.0.1:9/api/generate"  # nothing listens on port 9
TOOLS = {
    "generate_summary", "estimate_rul", "compare_batteries", "get_riskiest_battery",
    "get_healthiest_battery", "estimate_soh_for_battery_cycle", "get_charging_summary",
    "get_driving_score", "check_soh_deviation", "check_system_health",
    "simulate_what_if_scenarios",
}


@pytest.fixture(scope="module")
def agent():
    return DiscoveryMCPAgent(ollama_url=DEAD_OLLAMA)


# ------------------------------------------------------------- parsing LLM output
def test_parse_accepts_json_wrapped_in_prose():
    raw = 'Sure! ```json\n{"tool_name": "estimate_rul", "arguments": {"battery_id": "B0005"}}\n```'
    assert DiscoveryMCPAgent.parse_tool_selection(raw, TOOLS) == {
        "tool_name": "estimate_rul", "arguments": {"battery_id": "B0005"}}


@pytest.mark.parametrize("raw", [
    "I think you should use the summary tool.",            # no JSON at all
    '{"tool_name": "generate_summary", "arguments": {',   # truncated JSON
    '{"tool_name": "delete_everything", "arguments": {}}', # unknown tool
    '["generate_summary"]',                                 # wrong shape
    None,
])
def test_parse_rejects_unusable_output(raw):
    assert DiscoveryMCPAgent.parse_tool_selection(raw, TOOLS) is None


def test_parse_replaces_non_dict_arguments():
    raw = '{"tool_name": "generate_summary", "arguments": "B0005"}'
    assert DiscoveryMCPAgent.parse_tool_selection(raw, TOOLS)["arguments"] == {}


def test_filter_arguments_drops_invented_parameters():
    schema = {"properties": {"battery_id": {}, "cycle": {}}}
    assert DiscoveryMCPAgent.filter_arguments({"battery_id": "B0005", "battery": "x"}, schema) == {"battery_id": "B0005"}


# ----------------------------------------------------------------- fallback rules
@pytest.mark.parametrize("question, tool", [
    ("Compare B0005 and B0047", "compare_batteries"),
    ("Which battery is the riskiest?", "get_riskiest_battery"),
    ("Which battery is the healthiest?", "get_healthiest_battery"),
    ("Give me a measurement-based SOH estimate for B0018 cycle 60", "estimate_soh_for_battery_cycle"),
    ("What is the RUL of B0006?", "estimate_rul"),
    ("Show me the charging summary", "get_charging_summary"),
    ("How is my driving?", "get_driving_score"),
    ("Is there drift in the fleet?", "check_soh_deviation"),
    ("Any system warnings?", "check_system_health"),
    ("What if it is very cold?", "simulate_what_if_scenarios"),
    ("How healthy is my battery?", "generate_summary"),
    ("These are the rules of the game", "generate_summary"),  # 'rules' must not match RUL
])
def test_rule_based_fallback(question, tool):
    assert DiscoveryMCPAgent.rule_based_tool(question, TOOLS) == tool


# ------------------------------------------------------------ selection behaviour
def test_unreachable_ollama_falls_back_instead_of_crashing(agent):
    with pytest.raises(LLMUnavailableError):
        agent._call_ollama("ping")
    tool, args, method = agent.select_tool("Which battery is the riskiest?", "", TOOLS)
    assert (tool, args, method) == ("get_riskiest_battery", {}, "rule_based_fallback")


def test_invalid_llm_answer_is_retried_then_falls_back():
    a = DiscoveryMCPAgent(ollama_url=DEAD_OLLAMA)
    answers = iter(["no json here", '{"tool_name": "made_up_tool"}'])
    a.ask_gemma_for_tool = lambda q, t: next(answers)
    assert a.select_tool("Compare B0005 and B0047", "", TOOLS)[2] == "rule_based_fallback"


def test_valid_answer_on_retry_is_used():
    a = DiscoveryMCPAgent(ollama_url=DEAD_OLLAMA)
    answers = iter(["garbage", '{"tool_name": "estimate_rul", "arguments": {}}'])
    a.ask_gemma_for_tool = lambda q, t: next(answers)
    assert a.select_tool("anything", "", TOOLS) == ("estimate_rul", {}, "llm")


def test_ask_never_raises_when_the_mcp_server_cannot_start():
    a = DiscoveryMCPAgent(ollama_url=DEAD_OLLAMA)
    a.server_params.args = ["-c", "import sys; sys.exit(1)"]   # server dies immediately
    answer = a.ask("How healthy is B0005?")
    assert answer.startswith("⚠️ The agent could not complete this request")
    assert a.last_selected_tool is None


def test_explanation_falls_back_to_raw_tool_result(agent):
    text = agent.explain_or_raw("How healthy is B0005?", "generate_summary", '{"predicted_soh": 81.2}')
    assert "not available" in text and '{"predicted_soh": 81.2}' in text


# ------------------------------------------------------ end to end (needs MCP v1)
def test_ask_without_ollama_still_answers_with_tool_result(agent):
    pytest.importorskip("mcp.server.fastmcp", reason="needs mcp v1 (pip install 'mcp<2')")
    answer = agent.ask("Which battery is the riskiest?")
    assert agent.last_selected_tool == "get_riskiest_battery"
    assert agent.last_selection_method == "rule_based_fallback"
    assert "riskiest battery is" in answer
