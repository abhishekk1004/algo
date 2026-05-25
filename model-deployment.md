# Model Deployment & MLOps

Overview
- Deploying models into production includes packaging, serving, monitoring, and retraining workflows.

Important subtopics
- Serving (REST/gRPC), containerization, orchestration (Kubernetes)
- CI/CD, model versioning, feature stores
- Monitoring: data drift, performance, logging, alerts

Key notes
- Automate tests for data schema and model outputs before deployment.
- Track model and data versions for reproducibility and rollback.

Quick example (REST serving)
- Wrap a trained model in a Docker container exposing a `/predict` endpoint using FastAPI.

Mermaid pipeline
```mermaid
flowchart LR
  A[Train model] --> B[Package (Docker)]
  B --> C[Deploy to serving infra]
  C --> D[Monitor & log]
  D --> E[Retrain if drift detected]
```

Notes on images
- Add architecture diagram at `images/ml_deployment_arch.png`.
