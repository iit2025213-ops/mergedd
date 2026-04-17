# M5 Supreme: Decagon Ensemble & Hybrid Relational Forecasting
**Targeting < 0.5 WRMSSE on the Walmart M5 Challenge**

## 🧠 Project Vision
M5 Supreme is an advanced forecasting engine designed to crack the 0.5 WRMSSE barrier. This system treats the Walmart ecosystem as a **Dynamic Relational Manifold**, fusing the spatial depth of Graph Neural Networks (GNN) with the tabular precision of Gradient Boosting to capture both global macro-trends and local micro-interactions.

## 🏗️ The "Decagon" Architecture
The core of this system is the **Decagon Ensemble**, a modular manifold consisting of 10 distinct neural entities:

1.  **H-GNN (Hierarchical):** Models the structural physics from Item to State.
2.  **C-GNN (Behavioral):** Learns basket-level correlations (items that move together).
3.  **Graphormer:** Global attention-based relational modeling.
4.  **SigGNN (Path Geometry):** Captures demand "curvature" via Log-Signatures (Rough Path Theory).
5.  **ZI-GNN (Zero-Inflation):** Specifically architected for intermittent demand patterns.
6.  **E-GNN (Economic):** Maps log-price elasticity and cross-promotional effects.
7.  **CalGNN (Temporal):** Learns SNAP and holiday phase-shifts via Fourier encodings.
8.  **FlowGNN (Supply Chain):** Directed graph modeling of stock-out propagation (DC to Store).
9.  **VAT-GNN (Adversarial):** Built via Virtual Adversarial Training for robustness against shocks.
10. **Meta-Blender:** An attention-gated "10th Brain" that dynamically weights expert opinions.

### Hybrid Fusion Layer
Post-GNN, the system employs a **60/20/20 Fusion**:
- **60% GNN Ensemble:** Relational and contextual depth.
- **20% LGBM (DART):** Tabular generalization and trend stability.
- **20% XGBoost:** High-depth interaction specialist for non-linear patterns.

## 🌪️ Chaos Engineering & Robustness (Hawkes Process)
To ensure the forecasting pipeline can withstand real-world supply chain cascading failures, M5 Supreme is integrated with a state-of-the-art constraint-driven **Chaos Engineering Framework**. 

Rather than basic random noise (i.i.d.), failures are injected dynamically into the batch stream using a temporal **Hawkes Process**:
$$\lambda(t) = \mu + \sum_{t_k < t} \alpha \cdot e^{-\beta(t - t_k)}$$

This mathematically simulates "cascading failures" (e.g., if one store goes offline, nearby distribution nodes are statistically more likely to fail). The architecture strictly maintains the subcritical branching ratio ($\alpha/\beta < 1$) to prevent infinite failure loops.

Robustness ($R$) is parameterized and logged alongside validation: 
$$R(\mu, \alpha, \beta) = \frac{\text{WRMSSE}_{\text{clean}}}{\text{WRMSSE}_{\text{chaos}}}$$

## 📂 Repository Structure
```text
├── checkpoints/        # EMA Weights and model artifacts
├── configs/            # Centralized YAMLs for Model, Train, Data, and Boosting
├── data/               # Tiered storage: Raw (CSV), Processed (PT), Graphs (PT)
├── logs/               # Telemetry, Expert Trust Weights, and Audit Trails
├── notebooks/          # Diagnostic Suite (Topology, Signatures, Error Audits)
├── scripts/            # Orchestrators for Preprocessing, Graph Gen, and Training
├── src/
│   ├── models/         # Expert architectures and the Decagon Orchestrator
│   ├── boosting/       # LGBM/XGB Expert implementations
│   ├── engine/         # High-throughput data loaders and trainers
│   ├── chaos/          # Hawkes Process, Chaos Kong (Stores), Chaos Monkey (Data)
│   ├── pipeline/       # Chaos evaluation runners and datastore
│   └── utils/          # Graph Builders and Vectorized WRMSSE Metrics
├── analysis/           # Post-chaos robustness analyzers (R surface generation)
├── main.py             # Main entry point (w/ --hawkes-augmentation)
└── README.md
🛠️ Execution Pipeline
The project is designed as a 4-Phase Orchestration:

**Phase 1: Information Alignment** - `preprocess.py` converts raw CSVs into vectorized Parquet/PT formats.

**Phase 2: Topological Construction** - `generate_graphs.py` builds the 10 graph views.

**Phase 3: Expert Convergence & Chaos Injection** 
Run standard pipeline:
`python main.py`
Run with subcritical cascading chaos:
`python main.py --hawkes-augmentation`

**Phase 4: Hybrid Calibration** - Boosting experts are fitted to residuals for final precision tuning.

🧪 Research Diagnostics
01_Graph_Audit: Analyzes spectral gap and over-smoothing risks.

02_Signature_Analysis: Visualizes Path Geometry in signatory space.

03_Hierarchical_Audit: Deconstructs WRMSSE across all 12 hierarchical levels.

💻 Hardware Requirements
Optimal: NVIDIA A100 (80GB) or H100.

Minimum: 24GB VRAM (requires reducing hidden_dim and max_bin in configs).

Optimizations: NCCL P2P, Pinned Memory, and float32/bfloat16 precision.