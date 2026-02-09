## FILE: mentalsurvey.py

####### IMPORTING THE PYTHON LIBRARIES #######
##############################################

import pandas as pd
from pandas import DataFrame
from pandasgui import show
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


####### PHASE 1 - UNDERSTANDING THE DOCUMENT #######
####################################################

## Importing and opening of the Mental Health Survey file with Pandas
survey_data = pd.read_csv("./mental-heath-in-tech-2016_20161114.csv")

#print(survey_data.shape)                    # (row = 1433 and columns = 63)
#print (survey_data.dtypes.value_counts())   # The data types of the columns (object = 56, int64 = 4, and float64 = 3)
#gui = show(survey_data)                     # Shows the content of the data graphically


####### PHASE 2 - DATA CLEANING ########
########################################

## Block 1: Separating the features

## Block 1A: Identifying the numerical columns
numerical_columns = survey_data.select_dtypes(include=["int64", "float64"]).columns

#print(survey_data[numerical_columns])                   # The first 10 rows can also be outputted with ".head(10)" or the last with "tail(10)"
#gui = show(survey_data[numerical_columns])              # Graphically shows only the numeric columns in the data

## Block 1B: Identifying the categorical columns
## Note: The categorical columns contain mixed types in the same column, NaN or missing
## values and Invisible characters or whitespace: Thus, the data must be cleaned.
categorical_columns = survey_data.select_dtypes(include = "object").columns

#print(survey_data[categorical_columns])                  # The first 10 rows can also be outputted with ".head(10)" or the last with "tail(10)"
#gui = show(survey_data[categorical_columns])             # Graphically shows only the categorical columns in the data

## BLock 2: Handling the missing value

## Block 2A: Handing the missing Numerical values with median
survey_data[numerical_columns] = survey_data[numerical_columns].fillna(survey_data[numerical_columns].median())

## Block 2B: Replacing missing Categorical values with unknown
for columns in categorical_columns:
    survey_data[columns] = survey_data[columns].fillna("Unknown")            # This replaces the missing values
    survey_data[columns] = survey_data[columns].astype(str).str.strip()      # This convert the values to string and remove spaces

#gui = show(survey_data[categorical_columns])                                 # Current state of the categorical columns

## Block 2C: Normalizing gender and yes or no noise
survey_data["What is your gender?"] = (
    survey_data["What is your gender?"]
    .str.lower()
    .replace({
        "m": "male",
        "man": "male",
        "female ": "female",
        "f": "female",
        "woman": "female"
    })
)

#gui = show(survey_data)                     # Show the new processed data


####### PHASE 3 - FEATURE ENGINEERING #######
#############################################

## Block 1: Binary encoding (Yes/No → 1/0)
binary_map = {"Yes":1, "No":0}

for col in categorical_columns:
    if set(survey_data[col].unique()).issubset({"Yes","No","Unknown"}):
        survey_data[col] = survey_data[col].map(binary_map).fillna(0)

## Block 2: One-Hot Encoding (survey answers)
## Encoding the data for the ML model while keeping GUI readable
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categories = pd.DataFrame(encoder.fit_transform(survey_data[categorical_columns]))
categorical_cols_encoded = pd.concat([survey_data.select_dtypes(include='number'), encoded_categories], axis=1)   # This is the Feature matrix

## Block 3: Setting up the input dataset for the ML model
X = pd.concat([survey_data[numerical_columns], categorical_cols_encoded], axis=1)


####### PHASE 4 - DIMENSIONALITY REDUCTION #######
##################################################

## Block 1: Standardising the dataset
X.columns = X.columns.astype(str)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## Block 2: Using PCA to reduce dataset complexity
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

X_pca_df = pd.DataFrame(
    X_pca,
    columns=["PCA1", "PCA2"]
)

#print(X_pca_df)
#gui = show(X_pca_df)

## Block 3: Explained variance
pca.explained_variance_ratio_.sum()


####### PHASE 5 - SETTING UP THE ML-MODEL FOR THE CLUSTERING: USING K-MEANS #######
###################################################################################

## Block 1: Choosing K (Elbow Method)
inertia = []
for k in range(2,10):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_pca)
    inertia.append(km.inertia_)

#print(inertia)
#gui = show(inertia)

plt.plot(range(2,10), inertia)
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
#plt.show()

## Block 2: Final clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_pca)
survey_data["Cluster"] = clusters

#print(survey_data["Cluster"])            ## This print out the first and last 5 rows of cluster each participant belongs to"
#gui = show(survey_data["Cluster"])       ## Graphically shows the cluster (0 to 3) each participant belongs to.

## Block 3: PCA cluster plot
colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green", 3: "tab:red"}

plt.figure(figsize=(8, 6))
for cluster_id in range(4):
    plt.scatter(
        X_pca[clusters == cluster_id, 0],
        X_pca[clusters == cluster_id, 1],
        label=f"Cluster {cluster_id}",
        color=colors[cluster_id]
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Employee Mental Health Clusters")
plt.legend()
plt.show()


####### PHASE 6 - CLUSTER INTERPRETATION #######
################################################

## Block 1: Cluster size
survey_data["Cluster"].value_counts()

#print(survey_data["Cluster"].value_counts())           ## Prints out the number of participants in each cluster
#gui = show(survey_data["Cluster"].value_counts())      ## Graphically shows the number of participants in each cluster

## Block 2: Cluster profiles
cluster_summary = survey_data.groupby("Cluster").mean(numeric_only=True)     ## The mean used for each cluster (0 to 3)

#print(cluster_summary)
#gui = show(cluster_summary)                                                  ## Shows the means of the clusters for 9 columns



####### THE END #######
#######################


# IU-International University of Applied Sciences
# Course Code: DLBDSMLUSL01
# Author: Gabriel Manu
# Matriculation ID: 9212512