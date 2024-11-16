# Clustering
In this section, you will find the code and comprehensive results related to Chapter 5: "Clustering Patients Based on Blood Glucose Level Measurements."

## Code
The code consists of three Jupyter notebooks, detailed below.

### PreProcess.ipynb
In this Jupyter notebook, data preparation is performed so that clustering algorithms can later be trained. The data is sourced from the file **Glucose_measurements.csv** within the **T1DiabetesGranada** dataset. Statistical variables are calculated for each time range and patient, forming a row in the dataset to be created.

### Clustering.ipynb 
In this Jupyter notebook, experimentation with clustering methods is conducted. The main tasks include handling outliers, estimating partition size, and executing each of the clustering algorithms.

## Evaluation_and_analysis.ipynb
In this Jupyter notebook, the evaluation process of the clustering performed is carried out. The performance of the internal evaluation metric (Silhouette Coefficient) is obtained, the evaluation procedure designed in this research is applied, and the necessary charts and results are generated for the most prominent outcomes, enabling clinical experts to select the best clustering result.

## Inputs
This folder contains the file resulting from the data preprocessing process performed in the Jupyter notebook **PreProcess.ipynb**, which is used as input in the Jupyter notebook **Clustering.ipynb**.

## Outputs
This folder contains all the charts and results from the clustering and evaluation process in a comprehensive manner.
