# MCP-Based EV Digital Twin Agent

An AI-powered EV Digital Twin platform for battery health analytics, predictive maintenance, fleet monitoring, and intelligent decision support.

## Features

* Battery State of Health (SOH) Prediction
* Remaining Useful Life (RUL) Estimation
* Battery Range Estimation
* What-if Scenario Simulation
* Charging Intelligence & Risk Analysis
* Driving Intelligence
* Predictive Maintenance Recommendations
* Fleet Monitoring Dashboard
* Data Drift Detection
* LLM-Based EV Analyst
* MCP Server Tools

## Architecture

```text
Dashboard
    ↓
EV Digital Twin Agent
    ↓
Battery Twin Engine
    ↓
Monitoring / Drift Detection / Maintenance
    ↓
MCP Server Tools
```

## Datasets

### NASA Battery Dataset

Used for battery degradation modeling and SOH prediction.

### EV Charging Patterns Dataset

Used for charging risk analysis and charging intelligence.

### EV Telemetry Dataset

Used for driving behavior and energy consumption analytics.

## MCP Tools

* predict_future_soh
* estimate_rul

## Dashboard Modules

* Battery Twin
* Fleet Analytics
* Charging Intelligence
* Driving Intelligence
* Predictive Maintenance
* Monitoring
* Drift Detection
* AI Assistant

## Installation

```bash
pip install -r requirements.txt
```

## Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Future Work

* Full MCP Tool-Calling Agent
* Multi-Battery Comparison Tools
* Hugging Face Deployment
* Advanced Fleet AI Analyst

```
```
