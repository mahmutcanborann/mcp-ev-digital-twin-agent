"""Tests for the core analytics modules, the MCP server and (optionally) the LLM agent.

Replaces the old print-only scripts in the project root. Tests that need an
external service skip themselves when it is not available:
  * MCP server test  -> needs the `mcp` package v1 (FastMCP)
  * agent test       -> needs a running Ollama with gemma3:4b
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.battery_knowledge_base import BatteryKnowledgeBase  # noqa: E402
from src.battery_twin import BatteryDigitalTwin  # noqa: E402
from src.charging_analyzer import ChargingAnalyzer  # noqa: E402
from src.drift_detector import DataDriftDetector  # noqa: E402
from src.driving_analyzer import DrivingAnalyzer  # noqa: E402
from src.fleet_anomaly_detector import FleetAnomalyDetector  # noqa: E402
from src.monitoring_engine import MonitoringEngine  # noqa: E402
from src.predictive_maintenance import PredictiveMaintenanceEngine  # noqa: E402

HEALTH_STATES = {"Excellent", "Good", "Moderate", "Degraded"}


# ----------------------------------------------------------------- battery twin
@pytest.fixture(scope="module")
def twin():
    return BatteryDigitalTwin()


def test_twin_soh_is_a_percentage_and_degrades_with_cycles(twin):
    early = twin.predict_soh("B0005", cycle=10, ambient_temperature=24)
    late = twin.predict_soh("B0005", cycle=160, ambient_temperature=24)
    assert 0 < late < early <= 110


def test_twin_summary_is_internally_consistent(twin):
    s = twin.generate_summary("B0005", cycle=100, ambient_temperature=24, nominal_range_km=500)
    assert s["health_status"] == twin.get_health_status(s["predicted_soh"])
    assert s["estimated_range_km"] == pytest.approx(500 * s["predicted_soh"] / 100, abs=0.01)
    assert isinstance(s["remaining_cycles_until_80_soh"], int)


def test_twin_health_thresholds(twin):
    assert [twin.get_health_status(v) for v in (95, 85, 75, 50)] == ["Excellent", "Good", "Moderate", "Degraded"]


def test_twin_timeline_has_requested_steps(twin):
    timeline = twin.generate_timeline("B0005", start_cycle=0, end_cycle=150, step=50)
    assert [p["cycle"] for p in timeline] == [0, 50, 100, 150]
    assert all(p["health_status"] in HEALTH_STATES for p in timeline)


def test_what_if_scenarios_aggressive_is_not_better_than_normal(twin):
    sc = twin.simulate_what_if_scenarios("B0005", start_cycle=80, end_cycle=100)
    assert set(sc) == {"Normal", "Cold Weather", "Hot Weather", "Aggressive Usage"}
    assert sc["Aggressive Usage"][-1]["predicted_soh"] <= sc["Normal"][-1]["predicted_soh"]


# ------------------------------------------------------------ charging / driving
def test_charging_summary_ranges():
    c = ChargingAnalyzer()
    s = c.get_summary()
    assert 0 <= s["fast_charge_ratio"] <= 100
    assert s["avg_duration"] > 0
    assert sum(c.risk_distribution().values()) == len(c.df)
    assert all(r["charging_risk_level"] == "High" for r in c.high_risk_sessions())


def test_driving_score_and_style_are_consistent():
    d = DrivingAnalyzer()
    score = d.get_driving_score()
    assert 0 <= score <= 100
    expected = "Eco" if score >= 80 else "Normal" if score >= 60 else "Aggressive"
    assert d.get_driving_style() == expected


# --------------------------------------------------------------------- fleet
def test_drift_status_follows_standard_deviation_bands():
    det = DataDriftDetector()
    mean = det.reference_df["SOH"].mean()
    std = det.reference_df["SOH"].std()
    assert det.detect_soh_drift(mean)["drift_status"] == "Normal"
    assert det.detect_soh_drift(mean + 1.5 * std)["drift_status"] == "Warning"
    assert det.detect_soh_drift(mean + 3 * std)["drift_status"] == "Critical"


def test_fleet_anomalies_are_a_subset_of_the_fleet():
    det = FleetAnomalyDetector()
    anomalies = det.detect_anomalies()
    assert 0 < len(anomalies) < len(det.fleet_df)
    assert set(anomalies["battery_id"]) <= set(det.fleet_df["battery_id"])
    assert det.get_most_anomalous_battery()["battery_id"] in set(anomalies["battery_id"])


def test_monitoring_engine_status_matches_warnings():
    m = MonitoringEngine()
    h = m.check_system_health("B0005", cycle=150)
    n = len(h["warnings"])
    assert h["system_status"] == ("Healthy" if n == 0 else "Warning" if n <= 2 else "Critical")
    fleet = m.monitor_fleet()
    assert fleet["critical_batteries"] + fleet["warning_batteries"] + fleet["healthy_batteries"] == fleet["fleet_size"]


def test_predictive_maintenance_priority():
    r = PredictiveMaintenanceEngine().predict_actions("B0005", cycle=150)
    assert r["priority"] in {"Low", "Medium", "High"}
    assert isinstance(r["recommended_actions"], list)


def test_knowledge_base_only_returns_context_for_relevant_questions():
    kb = BatteryKnowledgeBase()
    assert "State of Health" in kb.retrieve_context("Why is the SOH low?")
    assert kb.retrieve_context("What time is it?") == ""


# ------------------------------------------------------------------ MCP server
def test_mcp_server_exposes_expected_tools():
    pytest.importorskip("mcp.server.fastmcp", reason="needs mcp v1 (pip install 'mcp<2')")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def list_tools():
        params = StdioServerParameters(command=sys.executable,
                                       args=[str(BASE_DIR / "mcp_servers" / "battery_soh_server.py")],
                                       cwd=str(BASE_DIR))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return {t.name for t in (await session.list_tools()).tools}

    tools = asyncio.run(list_tools())
    assert {"estimate_soh_for_battery_cycle", "estimate_soh_from_measurements",
            "generate_summary", "compare_batteries", "check_system_health"} <= tools
    assert not {"ask_ev_agent", "ask_llm_ev_agent"} & tools  # no agent-inside-a-tool


# ---------------------------------------------------------- agent (needs Ollama)
def _ollama_available() -> bool:
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate").replace("/api/generate", "/api/tags")
    try:
        return requests.get(url, timeout=2).ok
    except requests.RequestException:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama is not running")
def test_agent_selects_measurement_tool_and_reports_exact_numbers():
    from agent.discovery_mcp_agent import DiscoveryMCPAgent
    agent = DiscoveryMCPAgent()
    answer = agent.ask("Give me a measurement-based SOH estimate for B0018 cycle 60")
    assert agent.last_selected_tool == "estimate_soh_for_battery_cycle"
    assert "Measured SOH: 85.18%" in answer
