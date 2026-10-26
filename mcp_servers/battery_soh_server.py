import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from mcp.server.fastmcp import FastMCP

from src.battery_twin import BatteryDigitalTwin
from src.charging_analyzer import ChargingAnalyzer
from src.driving_analyzer import DrivingAnalyzer
from agent.ev_agent import EVDigitalTwinAgent
from src.monitoring_engine import MonitoringEngine
from src.drift_detector import DataDriftDetector
from src.soh_estimator import SOHEstimator, VALIDATED_MAE_SOH_POINTS, health_status
mcp = FastMCP("EVBatteryDigitalTwinServer")

twin = BatteryDigitalTwin()
charging_analyzer = ChargingAnalyzer()
driving_analyzer = DrivingAnalyzer()
fleet_agent = EVDigitalTwinAgent()
monitoring_engine = MonitoringEngine()
drift_detector = DataDriftDetector()
soh_estimator = SOHEstimator()


@mcp.tool()
def estimate_soh_for_battery_cycle(
    battery_id: str,
    cycle: int
) -> dict:
    """Estimate battery State of Health (SOH) from MEASUREMENTS: the first 5 minutes
    of the recorded discharge curve of a NASA test cycle.

    The model used here was trained WITHOUT this battery, so the estimate is
    out-of-sample. The measured SOH is returned next to the estimate so the
    error is visible. Prefer this tool when the user asks for a measurement-based
    or validated SOH estimate, or how accurate the SOH estimate is.
    """
    return soh_estimator.estimate_for_battery_cycle(
        battery_id=battery_id,
        cycle=cycle
    )


@mcp.tool()
def estimate_soh_from_measurements(
    voltage_at_60s: float,
    voltage_at_300s: float,
    mean_discharge_current: float,
    start_temperature: float,
    ambient_temperature: float = 24
) -> dict:
    """Estimate SOH (%) of any battery cell from the first 5 minutes of a discharge:
    terminal voltage at 60 s and 300 s (V), mean discharge current (A, positive),
    cell temperature at the start (degC) and ambient temperature (degC).
    Does not need a battery ID. Typical error: ~4.8 SOH points on unseen batteries.
    """
    features = {
        "ambient_temperature": ambient_temperature,
        "dis_i_mean": abs(mean_discharge_current),
        "dis_v_at_60s": voltage_at_60s,
        "dis_v_at_300s": voltage_at_300s,
        "dis_T_start": start_temperature,
    }
    soh = soh_estimator.estimate_from_features(features)
    return {
        "estimated_soh": soh,
        "health_status": health_status(soh),
        "typical_error_soh_points": VALIDATED_MAE_SOH_POINTS,
        "method": "first 5 minutes of the discharge curve, leave-one-battery-out validated",
    }

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
def compare_batteries(
    question: str,
    cycle: int = 150,
    ambient_temperature: int = 24,
    nominal_range_km: float = 500
) -> str:
    return fleet_agent.compare_batteries(
        question=question,
        cycle=cycle,
        ambient_temperature=ambient_temperature,
        nominal_range_km=nominal_range_km
    )


@mcp.tool()
def get_riskiest_battery() -> str:
    return fleet_agent.get_riskiest_battery()


@mcp.tool()
def get_healthiest_battery() -> str:
    return fleet_agent.get_healthiest_battery()

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