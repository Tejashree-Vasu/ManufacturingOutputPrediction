# 🏭 Manufacturing Equipment Output Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning web application built with **Streamlit**, **Scikit-learn**, and **Plotly** to estimate hourly manufacturing equipment throughput (`Parts_Per_Hour`) using real-time machine telemetry, material grades, operational shifts, and environmental parameters[cite: 1, 2].

---

## 📌 Executive Summary

Industrial manufacturing environments rely heavily on continuous process optimization to prevent bottlenecks and unplanned downtime. This project applies a trained **Linear Regression Pipeline** to forecast production rates with statistical confidence intervals, detect underperforming equipment, and run dynamic parameter sensitivity simulations[cite: 2, 8].

### Key Performance Benchmarks
* **R² Score**: `0.9059` (Explains ~90.6% of the output variance)[cite: 2, 8]
* **Root Mean Squared Error (RMSE)**: `3.5046` parts/hour[cite: 2, 8]
* **Mean Absolute Error (MAE)**: `2.7158` parts/hour[cite: 2, 8]
* **Mean Squared Error (MSE)**: `12.2820`[cite: 2, 8]

---

## ✨ Core Features

* **🎯 Single-Unit Output Prediction**:
  * Real-time calculation based on 13 numerical process indicators and 4 operational categories[cite: 1].
  * Automated feature derivation for `Total_Cycle_Time` ($Cycle\_Time + Cooling\_Time$) and `Temperature_Pressure_Ratio` ($Injection\_Temp / Injection\_Pressure$)[cite: 2].
  * Visual tolerance verification flagging equipment underperformance against observed baselines[cite: 2].

* **📈 Interactive Sensitivity (What-If) Simulation**:
  * Dynamic parameter sweep (±5% to ±30%) to simulate how adjusting cycle times, pressures, or temperatures impacts production yield[cite: 2].
  * High-resolution Plotly line curves showing the baseline vs. forecasted output.

* **📂 Batch Equipment Scoring**:
  * Upload operational CSV datasets to predict throughput across multiple machines simultaneously[cite: 2].
  * Calculates residuals ($Actual - Predicted$) and flags anomalies exceeding operational thresholds[cite: 2].
  * Export scored datasets directly as CSV[cite: 2].

* **🌓 Responsive Dark / Light UI**:
  * Custom theme-aware SVG badges and dynamic CSS using Streamlit native variables for clear visibility in all browser themes.
  * Built-in fallback to pipeline coefficients ensuring zero downtime if model serialization artifacts are missing[cite: 1].

---

## 📊 Input Variables & Schema

| Category | Features | Description |
| :--- | :--- | :--- |
| **Process Settings** | `Injection_Temperature`, `Injection_Pressure`, `Cycle_Time`, `Cooling_Time`[cite: 1] | Machine thermal and pneumatic actuation settings. |
| **Material & Environment** | `Material_Viscosity`, `Ambient_Temperature`[cite: 1] | Factory climate and raw plastic viscosity properties. |
| **Machine & Crew** | `Machine_Age`, `Operator_Experience`, `Maintenance_Hours`[cite: 1] | Asset wear, technician tenure, and servicing duration. |
| **Efficiency Metrics** | `Efficiency_Score`, `Machine_Utilization`[cite: 1] | Runtime utilization and mechanical operational indices. |
| **Operational Context** | `Shift`, `Machine_Type`, `Material_Grade`, `Day_of_Week`[cite: 1] | Work schedule, machine classification, and raw material quality[cite: 1]. |

---

## 🛠️ Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)[cite: 1]
* **Data & Machine Learning**: Python 3.10+, NumPy, Pandas, Scikit-learn[cite: 1]
* **Visualization**: Plotly Graph Objects, Matplotlib[cite: 1]
* **Model Serialization**: `pickle`

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
cd <YOUR_REPOSITORY_NAME>
