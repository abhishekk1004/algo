# Causality & Causal Inference

Overview
- Causal inference studies cause-and-effect relationships, going beyond correlation.

Important subtopics
- Directed acyclic graphs (DAGs), do-calculus
- Identification strategies: randomization, instrumental variables, regression discontinuity
- Estimation: propensity scores, matching, difference-in-differences

Key notes
- State causal assumptions explicitly; use domain knowledge to build DAGs.
- Observational studies require careful confounder control and sensitivity analysis.

Quick example (treatment effect)
- Estimate average treatment effect using propensity score matching between treated and control groups.

Mermaid pipeline
```mermaid
flowchart LR
  A[Domain knowledge] --> B[Build DAG]
  B --> C[Choose identification strategy]
  C --> D[Estimate effect & test assumptions]
```

Notes on images
- Add a sample DAG in `images/causal_dag.png`.
