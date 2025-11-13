# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:45:28 2025

@author: Freda Erinmwingbovo
"""

# Predicting Customer Computer Brand Preference

💻 **Project** — Predict which computer brand a customer prefers (Acer vs. Sony) from survey data.  
This repository contains a polished end-to-end machine learning pipeline and an interactive Streamlit app for predicting missing brand preferences and exploring results.

---

## 🔎 Overview

Some customers did not answer the brand-preference question in a survey. This project trains machine learning models on fully answered records, evaluates them, and uses the best-performing model to predict missing brand preferences. The output provides the sales team with a completed dataset and actionable regional and demographic insights.

**Key datasets**
- `CompleteResponses.csv` — 9,898 fully answered surveys (training & model development).
- `SurveyIncomplete.csv` — surveys with missing brand preferences (for prediction).

**Target**: `brand` — 0 = Acer, 1 = Sony.

---

## 🚀 Highlights

- End-to-end pipeline: Data description → EDA → Preprocessing → Modeling → Validation → Deployment.
- Models compared: Logistic Regression, Decision Tree, Random Forest.
- Final model: **Random Forest** (best test performance).
- Final deliverable: `SurveyIncomplete_Predictions.csv` and a regional summary table.
- Interactive Streamlit app to upload incomplete survey data and download predictions.

---

## 🧭 How to use

1. Clone the repository or copy this project folder to your machine.
2. Ensure the training data (`CompleteResponses.csv`) is in the same folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
