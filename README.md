# 🚦 Smart Traffic Digital Twin

A production-ready **Smart City Traffic Management Digital Twin** built with Python, Streamlit, YOLOv8, SQLite, NetworkX, Scikit-Learn, and PyDeck.

---

## 📌 Overview

This system simulates a real-world urban traffic management environment using an AI-powered Digital Twin. It detects vehicles from live video, predicts traffic conditions, optimizes signal timings, handles emergency vehicles, and visualizes everything on an interactive city map.

---

## 🗂️ Project Structure

```
traffic_digital_twin/
│
├── dashboard.py          ← Streamlit dashboard (main UI)
├── traffic_detector.py   ← YOLOv8 vehicle detection + tracking
├── prediction_engine.py  ← Scikit-Learn traffic prediction
├── signal_optimizer.py   ← Smart signal timing optimizer
├── simulation_engine.py  ← Multi-scenario traffic simulator
├── database.py           ← SQLite database module
├── network_model.py      ← NetworkX junction graph
├── arbitration.py        ← Multi-EV arbitration (merge/sequence/partial)
├── degradation.py        ← Graceful degradation hierarchy state machine
├── reid_handover.py      ← GPS-free cross-camera ReID handover (Novelty N2)
├── eta_predictor.py      ← Route prediction + uncertainty-aware ETA/UCB trigger (Novelty N3)
├── twin_engine.py        ← Digital twin decision engine: generate/simulate/select (Novelty N1)
│
├── latest_frame.jpg      ← Auto-saved by traffic_detector.py
├── traffic_twin.db       ← SQLite database (auto-created)
├── yolov8s.pt            ← YOLOv8 model weights
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the Dashboard

```bash
streamlit run dashboard.py
```

### 3. Run the Detector (separate terminal)

```bash
python traffic_detector.py --source output_ambulance_detection.mp4
```

> Use `--source 0` for webcam or `--no-window` for headless mode.

---

## ✨ Features

| Feature | Technology |
|---|---|
| Vehicle Detection & Tracking | YOLOv8 + OpenCV |
| Ambulance Detection | YOLOv8 class labels |
| Speed Estimation | Frame-to-frame displacement |
| Live Dashboard | Streamlit |
| Interactive City Map | PyDeck (ScatterplotLayer + HeatmapLayer) |
| Traffic Prediction | Scikit-Learn Polynomial Ridge Regression |
| Signal Optimization | Rule-based + Emergency Override |
| Junction Network | NetworkX with propagation |
| Database | SQLite |
| Analytics | Plotly + Pandas |
| Scenario Simulation | Python simulation engine |
| Multi-EV Arbitration | Union-find corridor clustering + priority scoring |
| Graceful Degradation | Deterministic state machine (edge autonomy / conservative hold) |
| Cross-Camera ReID Handover | Embedding distance + bearing/travel-time gated identity matching |
| Uncertainty-Aware ETA & Trigger | Route-probability + mean/std/UCB arrival bands (scipy) |
| Twin Decision Engine | Candidate generation → simulation → constrained selection (argmin J) |

---

## 🎮 Simulation Scenarios

| Scenario | Traffic Multiplier | Effect |
|---|---|---|
| Normal | 1.0× | Baseline city traffic |
| Rain Mode | 1.35× | Slower vehicles, higher density |
| Accident Mode | 1.8× | Major slow-down, ambulance response |
| Road Block | 2.2× | Closure, maximum diversion |
| Festival Traffic | 2.5× | Peak crowd movement |
| School Rush Hour | 1.6× | School zone activity |

---

## 🧠 Traffic Signal Rules

| Vehicle Count | Green Time |
|---|---|
| < 10 | 20 seconds |
| 10 – 20 | 40 seconds |
| 20 – 30 | 60 seconds |
| > 30 | 90 seconds |
| Ambulance | 90 seconds (override) |

---

## 🗄️ Database Schema

```sql
CREATE TABLE traffic (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT,
    vehicle_count   INTEGER,
    avg_speed       REAL,
    signal          TEXT,
    congestion      TEXT,
    ambulance       INTEGER
);

CREATE TABLE alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT,
    alert_type  TEXT,
    message     TEXT
);
```

---

## 📊 Dashboard Tabs

1. **📡 Live View & Map** — Camera feed + PyDeck 3D city map
2. **🔮 Prediction & Signals** — ML forecast + signal recommendations
3. **🌐 Network Graph** — NetworkX junction topology
4. **📊 Analytics** — Historical charts, distributions, tables
5. **🚨 Alerts** — Real-time and historical alert log
6. **⚙️ System Health** — Service status and system metrics
7. **🚑 Multi-EV & Resilience** — Multi-ambulance arbitration + degradation hierarchy (N5/N6)
8. **🧠 Twin Decision Engine** — Route prediction, ETA/UCB trigger, candidate schedules, ReID handover trace (N1/N2/N3)

---

## 🏗️ Architecture

```
Video Source ──► traffic_detector.py ──► SQLite DB ──► dashboard.py
                      │                                      │
                 YOLOv8 model                    simulation_engine.py
                 Speed estimation                prediction_engine.py
                 Ambulance flag                  signal_optimizer.py
                 Frame save                      network_model.py
```

---

## 🎓 Academic Use

This project is designed as a ** engineering project** demonstrating:
- Real-time computer vision (YOLOv8, OpenCV)
- Machine learning for time-series prediction (Scikit-Learn)
- Graph network modelling (NetworkX)
- Full-stack data pipeline (SQLite → Pandas → Streamlit)
- Smart city systems design

---

## 📋 License

For academic and educational use.
