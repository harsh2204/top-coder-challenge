# ACME Travel Reimbursement Calculator

This project aims to replicate ACME Corp's legacy travel reimbursement calculation system using modern machine learning techniques with tinygrad.

## Problem Description

The system calculates travel reimbursements based on three inputs:
- Trip duration (in days)
- Miles traveled
- Total receipts amount

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `model.py`: Contains the neural network architecture using tinygrad
- `train.py`: Training script for the reimbursement calculator
- `utils.py`: Helper functions for data processing and evaluation
- `public_cases.json`: Training dataset with known input/output pairs
- `private_cases.json`: Test dataset for final evaluation

## Usage

1. Train the model:
```bash
python train.py
```

2. Evaluate the model:
```bash
python evaluate.py
```

## Performance Metrics

The model aims to achieve:
- 99% accuracy on public test cases
- High accuracy on private test cases
- Minimal deviation from expected reimbursement amounts
