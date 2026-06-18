import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from mcp.server.fastmcp import FastMCP

from src.battery_twin import BatteryDigitalTwin
from src.charging_analyzer import ChargingAnalyzer
from src.driving_analyzer import DrivingAnalyzer
from agent.ev_agent import EVDigitalTwinAgent
from agent.llm_ev_agent import LLMEVDigitalTwinAgent
from src.monitoring_engine import MonitoringEngine
from src.drift_detector import DataDriftDetector
mcp = FastMCP("EVBatteryDigitalTwinServer")

twin = BatteryDigitalTwin()
charging_analyzer = ChargingAnalyzer()
driving_analyzer = DrivingAnalyzer()
ev_agent = EVDigitalTwinAgent()
llm_agent = LLMEVDigitalTwinAgent()
base_ev_agent = EVDigitalTwinAgent()
monitoring_engine = MonitoringEngine()
drift_detector = DataDriftDetector()

@mcp.tool()
def predict_soh(
    battery_id: str,
    cycle: int,
    ambient_temperature: int = 24
) -> float:
    return twin.predict_soh(
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=ambient_temperature
    )


@mcp.tool()
def estimate_rul(
    battery_id: str,
    current_cycle: int,
    threshold: float = 80,
    ambient_temperature: int = 24
) -> int:
    return twin.estimate_rul(
        battery_id=battery_id,
        current_cycle=current_cycle,
        threshold=threshold,
        ambient_temperature=ambient_temperature
    )


@mcp.tool()
def generate_summary(
    battery_id: str,
    cycle: int,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> dict:
    return twin.generate_summary(
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )


@mcp.tool()
def simulate_what_if_scenarios(
    battery_id: str,
    start_cycle: int,
    end_cycle: int,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> dict:
    return twin.simulate_what_if_scenarios(
        battery_id=battery_id,
        start_cycle=start_cycle,
        end_cycle=end_cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )

@mcp.tool()
def get_charging_summary() -> dict:
    return charging_analyzer.get_summary()


@mcp.tool()
def get_charging_risk_distribution() -> dict:
    return charging_analyzer.risk_distribution()


@mcp.tool()
def get_high_risk_charging_sessions() -> list:
    return charging_analyzer.high_risk_sessions()

@mcp.tool()
def get_driving_energy_summary() -> dict:
    return driving_analyzer.get_energy_summary()


@mcp.tool()
def get_driving_score() -> float:
    return driving_analyzer.get_driving_score()


@mcp.tool()
def get_driving_style() -> str:
    return driving_analyzer.get_driving_style()

@mcp.tool()
def ask_ev_agent(
    question: str,
    battery_id: str = "B0005",
    cycle: int = 150,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> str:
    return ev_agent.ask(
        question=question,
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )

@mcp.tool()
def ask_llm_ev_agent(
    question: str,
    battery_id: str = "B0005",
    cycle: int = 150,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> str:
    return llm_agent.ask(
        question=question,
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )
@mcp.tool()
def compare_batteries(
    question: str,
    cycle: int = 150,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> str:
    return base_ev_agent.compare_batteries(
        question=question,
        cycle=cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )


@mcp.tool()
def get_riskiest_battery() -> str:
    return base_ev_agent.get_riskiest_battery()


@mcp.tool()
def get_healthiest_battery() -> str:
    return base_ev_agent.get_healthiest_battery()

@mcp.tool()
def check_system_health(
    battery_id: str = "B0005",
    cycle: int = 150,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> dict:
    return monitoring_engine.check_system_health(
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )

@mcp.tool()
def detect_soh_drift(
    current_soh: float
) -> dict:
    return drift_detector.detect_soh_drift(
        current_soh=current_soh
    )

if __name__ == "__main__":
    mcp.run()