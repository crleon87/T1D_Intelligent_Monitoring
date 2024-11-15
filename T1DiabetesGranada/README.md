# Usage Notes
## Graphs and Stats.ipynb
The Jupyter Notebook generates tables, graphs and statistics for a better understanding of the dataset. It has four main sections, one dedicated to each file in the dataset. In addition, it has useful functions such as calculating the patient age, deleting a patient list from a dataset file and leaving only a patient list in a dataset file.

# Code availability
The dataset was generated using some custom code located in the folder "Code availability". The code is provided as Jupyter Notebooks and was used to conduct tasks such as data curation and transformation, and variables extraction.

## Original_patient_info_curation.ipynb
In the Jupyter Notebook is preprocessed the original file with patient data. Mainly irrelevant rows and columns are removed, and the sex variable is recoded.

## Glucose_measurements_curation.ipynb
In the Jupyter Notebook is preprocessed the original file with the continuous glucose level measurements of the patients. Principally rows without information or duplicated rows are removed and the variable with the timestamp is transformed into two new variables, measurement date and measurement time.

## Biochemical_parameters_curation.ipynb
In the Jupyter Notebook is preprocessed the original file with patient data of the biochemical tests performed on patients to measure their biochemical parameters. Mainly irrelevant rows and columns are removed and the variable with the name of the measured biochemical parameter is translated.

## Diagnostic_curation.ipynb
In the Jupyter Notebook is preprocessed the original file with patient data of the diagnoses of diabetes mellitus complications or other diseases that patients have in addition to T1D.

## Get_patient_info_variables.ipynb
In the Jupyter Notebook it is coded the feature extraction process from the files **Glucose_measurements.csv**, **Biochemical_parameters.csv** and **Diagnostics.csv** to complete the file **Patient_info.csv**. It is divided into six sections, the first three to extract the features from each of the mentioned files and the next three to add the extracted features to the resulting new file.
