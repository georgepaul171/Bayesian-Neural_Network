# Bayesian Emissions Risk Framework

Welcome to the Bayesian Emissions Risk Framework repository. This project supports George Paul's MSc research and prototype development for **Intelligent Carbon Risk Management** using Bayesian machine learning. The goal is to provide **traceable, probabilistic insights into embedded carbon emissions across global supply chains**, aligned with Apple 2030 and compliance mandates like **CA SB 253**.

## Project Objectives

- Develop a **Bayesian Neural Network (BNN)** to estimate **supplier-specific emissions** with **uncertainty quantification**
- Identify and flag **high-risk emitters** for traceability and audit purposes
- Simulate **regulatory audit workflows**, including compliance gaps
- Provide an **internal reporting dashboard** with interactive visualizations
- Integrate real or synthetic structured datasets for supplier emissions across scopes

---

## CA SB 253 Compliance Simulation

I implement a **Bayesian audit simulator** aligned with California’s SB 253 regulatory framework. Features:

- Tracks **Scope 1–3** emissions traceability
- Flags **non-standard disclosures** and **data incompleteness**
- Scores suppliers for **audit risk tiers** (Low, Medium, High)
- Simulates **audit response times** and **required documentation**

> The audit simulator ingests synthetic Apple-style supplier data, evaluates probabilistic emissions, and provides a compliance dashboard.

---

## Synthetic Apple-style Dataset

Inspired by Apple’s Environmental and Supply Chain Progress Reports (2023–24), the dataset includes:

| Field                          | Description                              |
|--------------------------------|------------------------------------------|
| `supplier_id`                 | Unique ID per supplier                   |
| `region`                      | Region of operations                     |
| `scope1_emissions`           | Direct GHG emissions                     |
| `scope2_emissions`           | Electricity-related GHGs                 |
| `scope3_upstream`            | Purchased goods, logistics               |
| `renewable_pct`              | Share of renewable electricity           |
| `audit_score`                | Historical audit compliance rating       |
| `reporting_gaps`             | # of missing or delayed disclosures      |


The second dataset, located in the 'synthetic-data-apple' folder, is a similar **synthetic dataset** with 10,000 rows.

This synthetic dataset was designed to simulate supplier-level emissions data aligned with Apple’s supply chain sustainability objectives and regulatory frameworks like CA SB 253. It includes plausible features such as scope-based emissions, renewable energy use, audit timing, material sourcing, and data traceability indicators. While anonymised and generated, the dataset structure reflects the types of internal data Apple tracks through initiatives like the Supplier Clean Energy Program and the Environmental Progress Reports. This prototype enables rigorous Bayesian modeling and uncertainty analysis, preparing the model for seamless integration with internal systems and audit workflows.

---

## Contributors

- **George Paul** – MSc Data Science, University of Bath  
- **Apple Sustainability Reports** – Inspiration for supplier profiling

---

## License

MIT License. See `LICENSE` for details.

> For questions or collaborations, please open an issue or contact George directly.
