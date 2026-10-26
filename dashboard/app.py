import json
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
from src.monitoring_engine import MonitoringEngine
from src.soh_deviation import FleetSOHDeviationChecker
from src.fleet_anomaly_detector import FleetAnomalyDetector
from agent.discovery_mcp_agent import DiscoveryMCPAgent
from src.soh_estimator import SOHEstimator, VALIDATED_MAE_SOH_POINTS
st.set_page_config(
    page_title="EV Digital Twin Platform",
    page_icon="🔋",
    layout="wide"
)

st.title("🔋 MCP-Based EV Digital Twin Platform")

twin = BatteryDigitalTwin()
driving_analyzer = DrivingAnalyzer()
maintenance_engine = PredictiveMaintenanceEngine()
monitoring_engine = MonitoringEngine()
deviation_checker = FleetSOHDeviationChecker()
fleet_anomaly_detector = FleetAnomalyDetector()
discovery_mcp_agent = DiscoveryMCPAgent()


@st.cache_resource
def load_soh_estimator():
    return SOHEstimator()


soh_estimator = load_soh_estimator()


@st.cache_data(show_spinner="Training models without this battery...")
def holdout_curve(selected_battery: str) -> pd.DataFrame:
    return soh_estimator.holdout_curve(selected_battery)


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

    # ---------------- measurement-based SOH, validated on an unseen battery ----------------
    st.subheader("🎯 Measurement-based SOH · validated on an unseen battery")
    st.caption(
        "Both estimates below come from models trained **without** the selected battery, "
        "i.e. what you would get on a brand-new cell. The new model only reads the first "
        "5 minutes of the discharge curve (voltage, current, temperature)."
    )

    validation = soh_estimator.estimate_for_battery_cycle(battery_id, cycle)

    if "error" in validation:
        st.warning(validation["error"])
    else:
        curve = holdout_curve(battery_id)
        shown_cycle = validation["cycle"]
        point = curve.loc[curve["cycle"] == shown_cycle].iloc[0]

        measured = validation["measured_soh"]
        new_estimate = validation["estimated_soh"]
        baseline_estimate = round(float(point["baseline_model"]), 2)

        mae_new = (curve["measurement_model"] - curve["measured_soh"]).abs().mean()
        mae_baseline = (curve["baseline_model"] - curve["measured_soh"]).abs().mean()

        v1, v2, v3 = st.columns(3)
        v1.metric(
            f"Measured SOH · cycle {shown_cycle}",
            f"{measured:.1f}%",
            help="Capacity measured in the NASA test, relative to the battery's maximum capacity."
        )
        v2.metric(
            "New model · first 5 min of discharge",
            f"{new_estimate:.1f}%",
            delta=f"{new_estimate - measured:+.1f} pts vs measured",
            delta_color="off",
            help=f"Typical error on unseen batteries: {VALIDATED_MAE_SOH_POINTS} SOH points."
        )
        v3.metric(
            "Original inputs · temperature + cycle",
            f"{baseline_estimate:.1f}%",
            delta=f"{baseline_estimate - measured:+.1f} pts vs measured",
            delta_color="off",
            help="Inputs of the original model, without the battery ID (which a new battery would not have)."
        )

        if "note" in validation:
            st.caption(f"ℹ️ {validation['note']}")

        fig_validation = go.Figure()
        fig_validation.add_trace(go.Scatter(
            x=curve["cycle"], y=curve["measured_soh"],
            name="Measured SOH", mode="lines",
            line=dict(color="#10b981", width=3),
            hovertemplate="%{y:.1f}%<extra></extra>"
        ))
        fig_validation.add_trace(go.Scatter(
            x=curve["cycle"], y=curve["measurement_model"],
            name=f"New model (MAE {mae_new:.1f})", mode="lines",
            line=dict(color="#3b82f6", width=2),
            hovertemplate="%{y:.1f}%<extra></extra>"
        ))
        fig_validation.add_trace(go.Scatter(
            x=curve["cycle"], y=curve["baseline_model"],
            name=f"Original inputs (MAE {mae_baseline:.1f})", mode="lines",
            line=dict(color="#9ca3af", width=2, dash="dash"),
            hovertemplate="%{y:.1f}%<extra></extra>"
        ))
        fig_validation.add_vline(
            x=shown_cycle, line_dash="dot", line_color="#6b7280",
            annotation_text=f"cycle {shown_cycle}", annotation_position="top"
        )
        fig_validation.add_hline(
            y=80, line_dash="dot", line_color="#ef4444",
            annotation_text="80% end-of-life", annotation_position="top right"
        )
        fig_validation.update_layout(
            xaxis_title="Cycle",
            yaxis_title="SOH (%)",
            hovermode="x unified",
            height=440,
            margin=dict(t=50, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0)
        )
        st.markdown(f"**{battery_id}: measured vs. estimated SOH over the battery's life**")
        st.plotly_chart(fig_validation, use_container_width=True, key="soh_validation_chart")

        fleet_text = ""
        results_path = BASE_DIR / "evaluation" / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())
            fleet = results["lobo"]["batteries_with_40plus_cycles"]
            fleet_new = fleet["First 5 min of discharge (production model)"]["mae"]
            fleet_old = fleet["Original project inputs (temperature + cycle)"]["mae"]
            n_batteries = results["dataset"]["long_batteries"]
            fleet_text = (
                f" Fleet-wide ({n_batteries} unseen batteries): **{fleet_new}** vs. **{fleet_old}**."
            )

        st.caption(
            f"Across all {len(curve)} cycles of {battery_id}: new model MAE **{mae_new:.2f}** "
            f"vs. original inputs **{mae_baseline:.2f}** SOH points.{fleet_text} "
            f"Details: `evaluation/results.md`."
        )

    st.divider()

    # ---------------- curve model (has seen this battery) ----------------
    st.subheader("📉 Curve model · trained on this battery")
    st.caption(
        "Reproduces this battery's own degradation curve from battery ID + cycle number; "
        "used for range, RUL and the simulations below."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Predicted SOH", f"{summary['predicted_soh']}%")
    col2.metric("Estimated Range", f"{summary['estimated_range_km']} km")
    rul_display = {
        "estimated": f"{summary['remaining_cycles_until_80_soh']} cycles",
        "already_below_threshold": "Reached",
        "not_projected": "Not projected",
    }[summary["rul_status"]]
    col3.metric("RUL to 80% SOH", rul_display, help=summary["rul_note"])
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
    st.subheader("📈 Digital Twin Timeline")

    timeline = twin.generate_timeline(
        battery_id=battery_id,
        start_cycle=0,
        end_cycle=200,
        step=50,
        ambient_temperature=temperature,
        nominal_range_km=nominal_range
    )

    timeline_df = pd.DataFrame(timeline)

    fig_timeline = px.line(
        timeline_df,
        x="cycle",
        y="predicted_soh",
        markers=True,
        title=f"{battery_id} Health Evolution",
        labels={
            "cycle": "Cycle",
            "predicted_soh": "SOH (%)"
        }
    )

    st.plotly_chart(
        fig_timeline,
        use_container_width=True,
        key="battery_timeline"
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
    st.caption(
        "⚠️ Scenario effects are **heuristic assumptions**, not calibrated from data: "
        "extra degradation of 0.02 / 0.04 / 0.08 SOH-% per cycle for cold / hot / aggressive use, "
        "on top of the curve model. Use this to explore sensitivity, not as a forecast."
    )

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

    st.divider()

    st.subheader("🚨 Fleet Anomaly Detection")

    anomalies_df = fleet_anomaly_detector.detect_anomalies()
    most_anomalous = fleet_anomaly_detector.get_most_anomalous_battery()

    if most_anomalous.get("status") == "No anomalies detected":
        st.success("No fleet anomalies detected.")
    else:
        a1, a2, a3 = st.columns(3)
        a1.metric("Most Anomalous Battery", most_anomalous["battery_id"])
        a2.metric("SOH", f"{most_anomalous['SOH']}%")
        a3.metric("Cycle", most_anomalous["cycle"])

        st.warning(most_anomalous["reason"])

        st.dataframe(anomalies_df)


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

    deviation = deviation_checker.check(summary["predicted_soh"])

    s1, s2, s3, s4 = st.columns(4)

    s1.metric("System Status", system_health["system_status"])
    s2.metric("Fleet Status", fleet_health["fleet_status"])
    s3.metric("Fleet Avg SOH", f"{fleet_health['average_soh']}%")
    s4.metric("Critical Batteries", fleet_health["critical_batteries"])

    st.subheader("📏 SOH Deviation from Fleet")
    st.caption(
        "Compares the selected battery's SOH with the fleet mean (z-score). "
        "This is an outlier check, not data-drift monitoring of input distributions."
    )

    dv1, dv2, dv3, dv4 = st.columns(4)

    dv1.metric("Fleet Mean SOH", f"{deviation['fleet_mean_soh']}%")
    dv2.metric("Current SOH", f"{deviation['current_soh']}%")
    dv3.metric("Deviation", f"{deviation['z_score']:+.2f} σ",
               help=f"{deviation['deviation_soh_points']:+.2f} SOH points from the fleet mean")
    dv4.metric("Status", deviation["status"])

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
        answer = discovery_mcp_agent.ask(
            question=question,
            battery_id=battery_id,
            cycle=cycle,
            ambient_temperature=temperature,
            nominal_range_km=nominal_range
        )
        selected_tool = discovery_mcp_agent.last_selected_tool
        selection_method = {
            "llm": "chosen by the LLM",
            "rule_based_fallback": "keyword fallback (LLM unavailable or invalid answer)",
            "multi_step": "multi-step assessment",
        }.get(discovery_mcp_agent.last_selection_method, "n/a")

        st.caption(f"🔧 Selected Tool: {selected_tool} · {selection_method}")

        st.info(answer)

    st.markdown("### Example Questions")
    st.write("• How healthy is my battery?")
    st.write("• Give me a measurement-based SOH estimate for B0018 cycle 60")
    st.write("• What maintenance actions do you recommend?")
    st.write("• Compare B0005 and B0047")
    st.write("• Which battery is the riskiest?")
    st.write("• Which battery is the healthiest?")