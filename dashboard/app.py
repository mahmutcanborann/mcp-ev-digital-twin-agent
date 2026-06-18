import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.battery_twin import BatteryDigitalTwin
from src.driving_analyzer import DrivingAnalyzer
from src.predictive_maintenance import PredictiveMaintenanceEngine
from agent.llm_ev_agent import LLMEVDigitalTwinAgent
from src.monitoring_engine import MonitoringEngine
from src.drift_detector import DataDriftDetector
from agent.tool_calling_ev_agent import ToolCallingEVAgent
from agent.mcp_tool_agent import MCPToolAgent

st.set_page_config(
    page_title="EV Digital Twin Platform",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 MCP-Based EV Digital Twin Platform")

twin = BatteryDigitalTwin()
driving_analyzer = DrivingAnalyzer()
maintenance_engine = PredictiveMaintenanceEngine()
ev_agent = LLMEVDigitalTwinAgent()
monitoring_engine = MonitoringEngine()
drift_detector = DataDriftDetector()
tool_calling_agent = ToolCallingEVAgent()
mcp_tool_agent = MCPToolAgent()


st.sidebar.header("Input Parameters")

battery_id = st.sidebar.selectbox(
    "Battery ID",
    ["B0005", "B0006", "B0007", "B0038", "B0042", "B0047"]
)

cycle = st.sidebar.slider(
    "Current Cycle",
    min_value=0,
    max_value=170,
    value=120
)

temperature = st.sidebar.selectbox(
    "Ambient Temperature (°C)",
    [4, 22, 24, 43, 44]
)

nominal_range = st.sidebar.number_input(
    "Nominal EV Range (km)",
    min_value=100,
    max_value=1000,
    value=500
)

summary = twin.generate_summary(
    battery_id=battery_id,
    cycle=cycle,
    ambient_temperature=temperature,
    nominal_range_km=nominal_range
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔋 Battery Twin",
    "📊 Fleet Analytics",
    "⚡ Charging & Driving",
    "📡 Monitoring",
    "🤖 Agent"
])


with tab1:
    st.header("🔋 Battery Digital Twin")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Predicted SOH", f"{summary['predicted_soh']}%")
    col2.metric("Estimated Range", f"{summary['estimated_range_km']} km")
    col3.metric("RUL to 80% SOH", f"{summary['remaining_cycles_until_80_soh']} cycles")
    col4.metric("Health Status", summary["health_status"])

    soh_value = float(summary["predicted_soh"])

    if soh_value >= 80:
        bar_color = "#16a34a"
    elif soh_value >= 70:
        bar_color = "#f59e0b"
    else:
        bar_color = "#dc2626"

    st.markdown(
        f"""
        <div style="margin-top:20px; margin-bottom:25px;">
            <div style="font-size:22px; font-weight:700; margin-bottom:8px;">
                🔋 Battery Health Score
            </div>
            <div style="background-color:#e5e7eb; border-radius:12px; height:28px; width:100%;">
                <div style="
                    background-color:{bar_color};
                    width:{soh_value}%;
                    height:28px;
                    border-radius:12px;
                    text-align:center;
                    color:white;
                    font-weight:700;
                    line-height:28px;">
                    {soh_value:.1f}%
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Future SOH Simulation")

    future = twin.simulate_future(
        battery_id=battery_id,
        start_cycle=cycle,
        end_cycle=cycle + 30,
        ambient_temperature=temperature,
        nominal_range_km=nominal_range
    )

    future_df = pd.DataFrame(future)

    st.line_chart(
        future_df.set_index("cycle")[["predicted_soh"]]
    )

    st.subheader("Future Range Simulation")

    st.line_chart(
        future_df.set_index("cycle")[["estimated_range_km"]]
    )

    st.subheader("Simulation Table")
    st.dataframe(future_df)

    st.divider()

    st.header("🧪 What-if Scenario Simulator")

    scenario_data = twin.simulate_what_if_scenarios(
        battery_id=battery_id,
        start_cycle=cycle,
        end_cycle=cycle + 50,
        ambient_temperature=temperature,
        nominal_range_km=nominal_range
    )

    scenario_rows = []

    for scenario_name, values in scenario_data.items():
        scenario_rows.extend(values)

    scenario_df = pd.DataFrame(scenario_rows)

    st.subheader("SOH under Different Scenarios")

    scenario_chart = scenario_df.pivot(
        index="cycle",
        columns="scenario",
        values="predicted_soh"
    )

    st.line_chart(scenario_chart)

    st.subheader("Estimated Range under Different Scenarios")

    range_chart = scenario_df.pivot(
        index="cycle",
        columns="scenario",
        values="estimated_range_km"
    )

    st.line_chart(range_chart)

    st.dataframe(scenario_df)


with tab2:
    st.header("📊 Fleet Analytics")

    fleet_path = BASE_DIR / "models" / "fleet_latest_status.csv"
    fleet_df = pd.read_csv(fleet_path)

    fleet_avg_soh = round(fleet_df["SOH"].mean(), 2)
    best_battery = fleet_df.sort_values("SOH", ascending=False).iloc[0]
    worst_battery = fleet_df.sort_values("SOH", ascending=True).iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric("Fleet Average SOH", f"{fleet_avg_soh}%")
    c2.metric("Best Battery", f"{best_battery['battery_id']} ({best_battery['SOH']:.2f}%)")
    c3.metric("Worst Battery", f"{worst_battery['battery_id']} ({worst_battery['SOH']:.2f}%)")

    st.subheader("Fleet SOH Distribution")

    fig_fleet = px.histogram(
        fleet_df,
        x="SOH",
        nbins=15,
        title="Fleet SOH Distribution"
    )

    st.plotly_chart(
        fig_fleet,
        use_container_width=True,
        key="tab2_fleet_soh_hist"
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🟢 Top 5 Healthy Batteries")
        healthy = fleet_df.sort_values("SOH", ascending=False).head(5)
        st.dataframe(healthy[["battery_id", "SOH", "cycle", "ambient_temperature"]])

    with c2:
        st.subheader("🔴 Top 5 Risky Batteries")
        risky = fleet_df.sort_values("SOH", ascending=True).head(5)
        st.dataframe(risky[["battery_id", "SOH", "cycle", "ambient_temperature"]])

    st.subheader("Fleet Temperature Distribution")

    temp_counts = fleet_df["ambient_temperature"].value_counts().reset_index()
    temp_counts.columns = ["temperature", "count"]

    st.bar_chart(
        temp_counts.set_index("temperature")
    )


with tab3:
    st.header("⚡ Charging Intelligence")

    charging_summary_path = BASE_DIR / "models" / "charging_summary.csv"
    risk_distribution_path = BASE_DIR / "models" / "charging_risk_distribution.csv"

    charging_summary = pd.read_csv(charging_summary_path)
    risk_distribution = pd.read_csv(risk_distribution_path)

    avg_duration = round(charging_summary.loc[0, "avg_duration"], 2)
    avg_soc_increase = round(charging_summary.loc[0, "avg_soc_increase"], 2)
    fast_charge_ratio = round(charging_summary.loc[0, "fast_charge_ratio"], 2)
    avg_risk_score = round(charging_summary.loc[0, "avg_risk_score"], 2)

    ch1, ch2, ch3, ch4 = st.columns(4)

    ch1.metric("Avg Charging Duration", f"{avg_duration} h")
    ch2.metric("Avg SOC Increase", f"{avg_soc_increase}%")
    ch3.metric("Fast Charging Usage", f"{fast_charge_ratio}%")
    ch4.metric("Avg Charging Risk Score", avg_risk_score)

    st.subheader("Charging Risk Distribution")

    st.bar_chart(
        risk_distribution.set_index("risk_level")["count"]
    )

    st.divider()

    st.header("🚗 Driving Intelligence")

    energy_summary = driving_analyzer.get_energy_summary()
    driving_score = driving_analyzer.get_driving_score()
    driving_style = driving_analyzer.get_driving_style()

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Average Speed", f"{energy_summary['avg_speed']} km/h")
    d2.metric("Average Power", f"{energy_summary['avg_power_kw']} kW")
    d3.metric("Driving Score", f"{driving_score}/100")
    d4.metric("Driving Style", driving_style)

    st.divider()

    st.header("🔧 Predictive Maintenance")

    maintenance = maintenance_engine.predict_actions(
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=temperature
    )

    m1, m2 = st.columns(2)

    m1.metric("Maintenance Priority", maintenance["priority"])
    m2.metric("Overall EV Twin Score", maintenance["overall_score"])

    st.subheader("Recommended Actions")

    for action in maintenance["recommended_actions"]:
        st.write(f"✅ {action}")


with tab4:
    st.header("📡 System Monitoring")

    system_health = monitoring_engine.check_system_health(
        battery_id=battery_id,
        cycle=cycle,
        ambient_temperature=temperature,
        nominal_range_km=nominal_range
    )

    fleet_health = monitoring_engine.monitor_fleet()

    drift_result = drift_detector.detect_soh_drift(
        summary["predicted_soh"]
    )

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("System Status", system_health["system_status"])
    s2.metric("Fleet Status", fleet_health["fleet_status"])
    s3.metric("Fleet Avg SOH", f"{fleet_health['average_soh']}%")
    s4.metric("Critical Batteries", fleet_health["critical_batteries"])

    st.subheader("📡 Data Drift Monitoring")

    dr1, dr2, dr3, dr4 = st.columns(4)

    dr1.metric("Reference Mean SOH", f"{drift_result['reference_mean_soh']}%")
    dr2.metric("Current SOH", f"{drift_result['current_soh']}%")
    dr3.metric("Drift Distance", drift_result["drift_distance"])
    dr4.metric("Drift Status", drift_result["drift_status"])

    st.subheader("Fleet Health Overview")

    c1, c2 = st.columns(2)

    c1.metric(
        "Worst Battery",
        fleet_health["worst_battery"]["battery_id"],
        f"{fleet_health['worst_battery']['soh']}%"
    )

    c2.metric(
        "Best Battery",
        fleet_health["best_battery"]["battery_id"],
        f"{fleet_health['best_battery']['soh']}%"
    )

    fleet_monitor_df = monitoring_engine.fleet_df

    st.subheader("Fleet SOH Distribution")

    fig_hist = px.histogram(
        fleet_monitor_df,
        x="SOH",
        nbins=15,
        title="Fleet SOH Distribution"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True,
        key="tab4_fleet_soh_histogram"
    )

    fig_fleet_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fleet_health["average_soh"],
            title={"text": "Fleet Average Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "white"},
                "steps": [
                    {"range": [0, 70], "color": "red"},
                    {"range": [70, 80], "color": "orange"},
                    {"range": [80, 100], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": fleet_health["average_soh"]
                }
            }
        )
    )

    st.plotly_chart(
        fig_fleet_gauge,
        use_container_width=True,
        key="tab4_fleet_average_health_gauge"
    )

    fig_battery_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=summary["predicted_soh"],
            title={"text": f"{battery_id} Battery SOH"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "white"},
                "steps": [
                    {"range": [0, 70], "color": "red"},
                    {"range": [70, 80], "color": "orange"},
                    {"range": [80, 100], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": summary["predicted_soh"]
                }
            }
        )
    )

    st.plotly_chart(
        fig_battery_gauge,
        use_container_width=True,
        key="tab4_selected_battery_soh_gauge"
    )

    worst_df = fleet_monitor_df.sort_values("SOH").head(5)
    best_df = fleet_monitor_df.sort_values("SOH", ascending=False).head(5)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🔴 Top 5 Critical Batteries")
        st.dataframe(worst_df[["battery_id", "SOH", "cycle"]])

    with c2:
        st.subheader("🟢 Top 5 Healthy Batteries")
        st.dataframe(best_df[["battery_id", "SOH", "cycle"]])

    st.subheader("System Warnings")

    if system_health["warnings"]:
        for warning in system_health["warnings"]:
            st.warning(warning)
    else:
        st.success("No system warnings.")


with tab5:
    st.header("🤖 EV Digital Twin Agent")

    question = st.text_input(
        "Ask a question about the EV digital twin",
        "How healthy is my battery?"
    )

    if question:
        answer = tool_calling_agent.ask(
            question=question,
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=temperature,
            nominal_range_km=nominal_range
        )

        st.info(answer)

    st.markdown("### Example Questions")
    st.write("• How healthy is my battery?")
    st.write("• What maintenance actions do you recommend?")
    st.write("• Compare B0005 and B0047")
    st.write("• Which battery is the riskiest?")
    st.write("• Which battery is the healthiest?")