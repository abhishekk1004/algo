# Probabilistic Methods & Bayesian AI

Overview
- Methods that model uncertainty explicitly using probability distributions and Bayesian inference.

Important subtopics
- Bayesian networks, graphical models
- Inference techniques: MCMC, variational inference
- Probabilistic programming (PyMC, Stan)

Key notes
- Useful when quantifying uncertainty is required (medical diagnosis, forecasting).

Quick example (Bayesian linear regression)
- Place priors on weights and compute posterior over parameters given data.

Mermaid pipeline
```mermaid
flowchart LR
  A[Priors] --> B[Likelihood]
  B --> C[Posterior (Bayes rule)]
  C --> D[Predictive distribution]
```

Notes on images
- Add a posterior distribution plot at `images/bayes_posterior.png`.
