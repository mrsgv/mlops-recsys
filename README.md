# mlops-recsys

MLOps Course End-Term Project — Recommendation System.

## Team
- TBD

## Overview
End-to-end MLOps pipeline for a recommendation system. Details TBD.

## Mandatory Components (checklist)
- [ ] Version Control (Git/GitHub)
- [ ] Data Pipeline — Airflow + (Spark / Kafka / Beam)
- [ ] Data Processing — cleaning, feature engineering, preprocessing
- [ ] Model Development — 2+ models, MLflow tracking
- [ ] Versioning — DVC
- [ ] Deployment — FastAPI + Docker
- [ ] Monitoring — logging + (Prometheus / drift / latency)
- [ ] CI/CD — GitHub Actions
- [ ] Documentation

## Folder Structure
```
airflow/dags/       Airflow DAG(s) for the data/training pipeline
src/data/           data ingestion + processing
src/models/         training + evaluation
src/api/            FastAPI serving app
data/               raw + processed data (DVC-tracked)
models/             trained model artifacts
tests/              unit tests
.github/workflows/  CI/CD
```

## Setup
TBD

## Running
TBD
