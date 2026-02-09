# Employee Mental Health Clustering Analysis

This repository contains a Python-based data analysis and machine learning pipeline for exploring patterns in
employee mental health survey data using unsupervised learning. The project focuses on data cleaning, feature
engineering, dimensionality reduction, and clustering to identify distinct groups of respondents based on their
survey responses.

The workflow is clearly structured into phases, making it suitable for academic coursework, data science learning,
and reproducible research.

## Project Overview

The goal of this project is to:\
Clean and preprocess a real-world mental health survey dataset\
Transform mixed-type survey responses into machine-learning-ready features\
Reduce high-dimensional data using Principal Component Analysis (PCA)\
Apply clustering algorithm to identify patterns among survey participants (K_means was used)\
Interpret and visualize the resulting clusters\

The dataset used is the Mental Health in Tech Survey (2016), which contains demographic, workplace, and mental
health–related responses from Kaggle (follow the link https://www.kaggle.com/datasets/omsi/mental-health-in-tech-2016)

## Project Structure and Workflow

The code is divided into clearly defined phases:\

🔹 Phase 1: Understanding the Dataset\
Loads the CSV file using Pandas\
Inspects dataset shape and column data types\
(Optional) Visual exploration using pandasgui\
Purpose: Gain an initial understanding of the dataset structure and data quality.\

🔹 Phase 2: Data Cleaning\
1. Feature Separation\
Numerical columns identified (int64, float64)\
Categorical columns identified (object)\

2. Handling Missing Values\
Numerical values: filled using the column median\
Categorical values: replaced with "Unknown"\
Whitespace removed and values converted to strings\

3. Data Normalization\
Gender responses normalized (for example, m, man → male; f, woman → female)\
Purpose: Ensure consistency, completeness, and usability of the data.\

🔹 Phase 3: Feature Engineering\
1. Binary Encoding\

2. One-Hot Encoding\
Categorical survey responses transformed using OneHotEncoder\
Unknown categories handled gracefully\

3. Feature Matrix Construction\
Numerical and encoded categorical features combined into a single dataset (X)\
Purpose: Convert survey data into a machine-learning-compatible format.\

🔹 Phase 4: Dimensionality Reduction (PCA)
Data standardized using StandardScaler\
Principal Component Analysis (PCA) applied\
Reduced to 2 principal components for visualization\
Explained variance ratio evaluated\
Purpose: Reduce dataset complexity while preserving meaningful variance.

🔹 Phase 5: Clustering with K-Means
1. Elbow Method\
K values from 2 to 9 evaluated\
Inertia plotted to identify optimal number of clusters

2. Final Model\
K-Means applied with 4 clusters\
Cluster labels assigned to each survey respondent

3. Visualization\
2D scatter plot of PCA components\
Each cluster displayed with a distinct color\
Purpose: Identify natural groupings in employee mental health responses.

🔹 Phase 6: Cluster Interpretation\
Cluster sizes calculated\
Mean values of numerical features computed per cluster

This allows:\
Comparison of mental health indicators across clusters\
Insight into dominant characteristics of each group

## Outputs and Visualizations

Elbow curve for K selection\
PCA scatter plot showing clusters\
Cluster size distribution\
Cluster-level numerical summaries

This project demonstrates:\
Practical data preprocessing for real-world datasets\
Feature encoding strategies for mixed data types\
Use of PCA for interpretability and visualization\
Application of unsupervised learning in social science research

It is suitable for:\
Data science coursework\
Machine learning demonstrations\
Exploratory analysis of mental health survey data

## Notes and Limitations

Clustering is exploratory and does not imply causation\
Interpretation depends heavily on feature selection and encoding\
PCA reduces interpretability of individual original features

**DISCLAIMER**_This project is intended for educational and academic use. Please refer to the original dataset source for data 
licensing and usage terms._



IU-International University of Applied Sciences\
 Course Code: DLBDSMLUSL01\
 Author: Gabriel Manu\
 Matriculation ID: 9212512\
