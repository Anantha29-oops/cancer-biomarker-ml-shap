# Cancer Biomarker Discovery Using Machine Learning and SHAP

## Project Overview

This project presents a Machine Learning and Explainable AI approach
for identifying potential cancer biomarker candidates from RNA-seq
gene-expression data.

The project uses TCGA cancer samples and GTEx normal tissue samples.

## Dataset

Source:
- The Cancer Genome Atlas (TCGA)
- Genotype-Tissue Expression (GTEx)
- UCSC Xena

The final modeling dataset contains:
- 4,372 samples
- 2,186 cancer samples
- 2,186 normal samples
- 37,978 gene features
- 2,000 selected genes

## Machine Learning Models

The following models were evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine
- XGBoost

## Explainable AI

SHAP-based feature interpretation was used to identify
genes that contributed strongly to model predictions.

## Results

XGBoost achieved:

Accuracy: 1.0000
Precision: 1.0000
Recall: 1.0000
F1-score: 1.0000
ROC-AUC: 1.0000

## Web Application

A Flask-based web application was developed to demonstrate
cancer/normal classification and gene-level contribution
information.

## Disclaimer

This project is intended for research and educational purposes.
The model is not a clinical diagnostic system. Genes identified
by the model are potential biomarker candidates and require
independent validation.

## Project Structure

...
