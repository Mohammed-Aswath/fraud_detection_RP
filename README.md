# Fraud Risk Manager

AI Risk Manager — Fraud Spike Detector for the Razorpay AI Buildathon, Track 02.

## Setup

```bash
# Install the locked Python environment and project dependencies
uv sync

# Retrieve versioned data and model artifacts (after the DVC remote is configured)
uv run dvc pull

# Run the test suite
uv run pytest
```

Copy `.env.example` to `.env` when environment-specific credentials are introduced. Do not commit `.env`.

## Reproducibility

Code is versioned on GitHub. Data and model artifacts are versioned with DVC using a DagsHub remote. Experiments are tracked with MLflow on DagsHub, so runs can be compared and reproduced from their configuration and artifacts.

## Problem Statement

_To be defined._

## Approach

_To be defined._

## Results

_To be defined._
