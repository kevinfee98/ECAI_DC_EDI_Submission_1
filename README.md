# Evolving ECAI Topics and Contributors

This repository contains all code, data, and visualisations for the paper **"Evolving ECAI Topics and Contributors: A Predictive Framework"**, submitted to the ECAI 2025 Doctoral Consortium Equality, Diversity, and Inclusion Competition.

## Repository Structure
- `1. Data_Extraction.ipynb`: Collects DOIs from DBLP and queries SemOpenAlex for contributor geography and topic metadata.
- `2. Visualise_Contributors.ipynb`: Explores country-level author participation statistics and produces bar chart summaries.
- `3. Build_PredictiveFramework.ipynb`: Defines the multivariate LSTM architecture used for temporal forecasting of topic popularity.
- `4. Topic_Prediction.ipynb`: Trains the predictive model on historical topic frequencies and generates ECAI 2025 forecasts.
- `5. DataVisualization_Notebook.ipynb`: Creates static and animated visualisations of topic evolution over time.
- `Data/`: Intermediate JSON artefacts used by the notebooks (countries, topic labels, yearly topic counts, predictions).
- `Visualisations/`: Saved figures that can be embedded in the manuscript or presentations.

## Prerequisites
- Python 3.9 or newer.
- Jupyter Notebook or JupyterLab.
- Recommended packages: `pandas`, `numpy`, `tensorflow`, `plotly`, `matplotlib`, `sympy`, `SPARQLWrapper`, `imageio`. Each notebook installs missing dependencies with `%pip install`, but pre-installing speeds up execution.

## Quick Start
1. Clone the repository and `cd` into it.
2. (Optional) Create and activate a virtual environment.
3. Launch Jupyter (`jupyter notebook` or `jupyter lab`).
4. Run the notebooks in numerical order (`1` through `5`). Several notebooks depend on artefacts written by earlier steps, so keeping the sequence avoids missing files.

## Data Outputs
- `Data/countries.json`: Country-level counts of authorship derived from SemOpenAlex.
- `Data/topic_label.json`: Mapping from topic URIs to human-readable labels.
- `Data/topic_year_counts.json`: Yearly frequency of topics across proceedings.
- `Data/ECAI_2025_Topic_Predictions.json`: Forecasted topic counts produced by the predictive framework.

## Citation
If you build upon this work, please cite the authors: Nicolò Donati[0009−0000−5673−5274], Kevin Fee[0009−0001−5062−4247], and
Verdiana Schena[0009−0009−1660−5462].
