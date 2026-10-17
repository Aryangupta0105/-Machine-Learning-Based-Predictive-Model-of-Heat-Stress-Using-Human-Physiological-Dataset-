# 🌡️ Non-Invasive Heat Stress Predictive Engine

[![Live Application](https://img.shields.io/badge/Demo-Live_on_Render-brightgreen?style=flat-square&logo=render)](https://heat-stress-ai.onrender.com)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20Gunicorn-lightgrey?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20GradientBoosting-orange?style=flat-square&logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

> An end-to-end computational physiology system and live clinical web dashboard that predicts continuous Core Body Temperature ($T_{\text{core}}$) non-invasively to detect heat stress risk before clinical onset.

🔗 **Live Web Application:** [https://heat-stress-ai.onrender.com](https://heat-stress-ai.onrender.com)

---
<img width="1383" height="907" alt="Screenshot (104)" src="https://github.com/user-attachments/assets/b0c2c9a7-e128-49bf-a072-d026028e45e6" />

## 📌 Overview

Monitoring internal core body temperature ($T_{\text{core}}$) in real time is critical for preventing heat-related illnesses such as exertional heat stroke and severe hyperthermia among industrial workers, athletes, and defense personnel. However, gold-standard internal thermometry methods (rectal, esophageal, or ingestible telemetry pills) are invasive, uncomfortable, and impractical for field operations.

This project delivers a **data-driven machine learning alternative** trained on human physiological exposure datasets. By measuring peripheral and hemodynamic responses, the system predicts core temperature with high accuracy and visualizes physiological status via an interactive, responsive medical dashboard.

---

## 📊 Key Highlights & Benchmark Results

The pipeline was benchmarked against multiple ensemble and linear regression architectures using cross-validated physiological parameters:

| Model Architecture | $R^2$ Score | Validation Status | Real-Time Latency |
| :--- | :--- | :--- | :--- |
| **Gradient Boosting Regressor (Deployed)** | **98.89%** | **Validated** | **< 15 ms** |
| Random Forest Regressor | 97.42% | Evaluated | ~22 ms |
| XGBoost Regressor | 98.15% | Evaluated | ~18 ms |
| Multiple Linear Regression | 84.10% | Baseline | < 5 ms |

---

## 🧠 Physiological Features Monitored

The model infers internal thermal states using 11 peripheral, autonomic, and hemodynamic features:

* **Mean Skin/Body Temperature (`Mean_Tbody`):** Weighted thermistor distribution across body surfaces.
* **Laser-Doppler Skin Perfusion (`LDF`) & Local Sweat Rate (`LSR`):** Peripheral microvascular dilation and active sudoromotor response.
* **Hemodynamic Metrics (`HR_initial`, `SBP`, `DBP`, `MAP`):** Heart rate, systolic/diastolic pressures, and mean arterial pressure reflecting cardiovascular heat strain.
* **Autonomic Activity (`SSNA`):** Skin sympathetic nerve activity quantifying thermoregulatory efferent drive.
* **Exposure & Fluid Loss (`Heat_exposure`, `Sweat_rate`, `change`):** Cumulative heat exposure duration and rapid dehydration volume metrics.

### Clinical Diagnostic Thresholds
* **Normothermic:** $T_{\text{core}} < 38.0^\circ\text{C}$ (Safe operating physiological state)
* **Hyperthermic (Heat Strain Alert):** $T_{\text{core}} \ge 38.0^\circ\text{C}$ (Elevated thermal risk; immediate mitigation required)

---

## 🏗️ System Architecture

```text
├── .python-version                # Pinned runtime environment (Python 3.10)
├── CLEAN_physiological_data.csv   # Structured physiological baseline dataset
├── model_gb.pkl                   # Pre-trained Gradient Boosting regression weights
├── app.py                         # Production Flask backend + dynamic self-healing fallback
├── requirements.txt               # Application dependencies
└── templates/
    └── index.html                 # Professional dark-mode clinical monitoring UI
