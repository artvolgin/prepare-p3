# PREPARE Challenge: Phase 3. Early Detection of AD/ADRD: A Robust, Fair, and Explainable Framework for Cognitive Score Prediction

This repository contains a comprehensive machine learning pipeline for cognitive score prediction across multiple countries. The repo includes data preprocessing, model training, and evaluation examples.

## Dependencies

The project requires Python 3.11.8 and the following setup:

1. Create and activate a conda environment:
   ```bash
   conda create --name prepare-submission python=3.11.8
   conda activate prepare-submission
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install project in editable mode:
   ```bash
   pip install -e .
   ```

## Required Data Files

The following files need to be downloaded and placed in the appropriate directories to run the pipeline:

### 1. India Data

- Harmonized LASI Core Data:
  - File: `../data/core/LASI/Harmonized LASI A.3_Stata/H_LASI_a3.dta`
  - Source: [Gateway to Global Aging - LASI](https://g2aging.org/lasi/download)

- Harmonized LASI-DAD:
  - File: `../data/hcap/LASI/Harmonized_LASI-DAD_Ver_B.1/H_LASI_DAD_b1.dta`
  - Source: [Gateway to Global Aging - LASI-DAD](https://g2aging.org/lasi-dad/)

### 2. US Data

- Harmonized HRS Core:
  - File: `../data/core/HRS/H_HRS_d_spss/H_HRS_d.sav`
  - Source: [HRS - Gateway Harmonized HRS](https://hrsdata.isr.umich.edu/data-products/gateway-harmonized-hrs)

- RAND HRS Longitudinal:
  - File: `../data/core/HRS/randhrs1992_2022v1_STATA/randhrs1992_2022v1.dta`
  - Source: [HRS - RAND HRS Longitudinal File 2022](https://hrsdata.isr.umich.edu/data-products/rand-hrs-longitudinal-file-2022)

- Harmonized HRS End of Life:
  - File: `../data/core/HRS/HarmonizedHRSEndOfLifeA/H_HRS_EOL_a.dta`
  - Source: [HRS - Gateway Harmonized HRS End of Life](https://hrsdata.isr.umich.edu/data-products/gateway-harmonized-hrs-end-life)

- HRS HCAP Harmonized Factor Scores:
  - File: `../data/hcap/HRS/HCAP-Harmonized-Factor-Scores/interim-HcapHarmoniz-305-v2-20230725-hrshcap.dta`
  - Source: [HRS - HCAP Measures Statistically Harmonized with Other HCAPs](https://hrsdata.isr.umich.edu/data-products/hrs-hcap-measures-statistically-harmonized-other-hcaps)

### 3. UK Data

- Harmonized ELSA Core:
  - File: `../data/core/ELSA/UKDA-5050-stata/stata/stata13_se/h_elsa_g3.dta`
  - Source: [UK Data Service - ELSA](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=5050)

- ELSA HCAP:
  - File: `../data/hcap/ELSA/spss/spss28/hcap_2018_harmonised_scores.sav`
  - Source: [UK Data Service - ELSA HCAP](https://beta.ukdataservice.ac.uk/datacatalogue/studies/study?id=9081)

### 4. Mexico Data

- Harmonized MHAS Core:
  - File: `../data/core/MHAS/H_MHAS_c2.sav`
  - Source: [MHAS - Gateway Harmonized MHAS (Version D)](https://www.mhasweb.org/DataProducts/HarmonizedData.aspx)

- Mexico HCAP:
  - File: `../data/hcap/Mex-Cog/Multi-Country Harmonized Factor Scores Mex-Cog 2016.dta`
  - Source: [MHAS - Gateway Harmonized 2016 Mex-Cog (Version A.2)](https://www.mhasweb.org/DataProducts/HarmonizedData.aspx)

**Note**: Access to these datasets may require registration and acceptance of data usage agreements with the respective institutions.

## Reproducing Results

To reproduce the analysis and results from the report:

1. Complete the Dependencies setup above
2. Download and organize the Required Data Files as specified
3. Run the following notebooks in sequence:
   - `notebooks/combine_datasets.ipynb`: Data integration and harmonization
   - `notebooks/run_specifications.ipynb`: Model training and evaluation
   - `notebooks/report_output.ipynb`: Results and visualization generation

For examples of applying this pipeline to new datasets, see `notebooks/usage_examples.ipynb`.

## Project Structure

```
prepare-phase-3/
├── data/                    # Data storage directory
├── models/                  # Trained model storage
├── notebooks/              # Jupyter notebooks for analysis
│   ├── combine_datasets.ipynb     # Data integration from multiple sources
│   ├── run_specifications.ipynb   # Model training specifications
│   └── report_output.ipynb       # Analysis and visualization
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── base_model.py     # Base class for all models
│   │   ├── catb_model.py     # CatBoost implementation
│   │   ├── lgbm_model.py     # LightGBM implementation
│   │   ├── xgbm_model.py     # XGBoost implementation
│   │   └── ensb_model.py     # Ensemble model implementation
│   ├── pipeline/          # Pipeline components
│   │   └── pipeline_runner.py # Main pipeline orchestration
│   ├── preprocess/        # Data preprocessing
│   │   └── data_preprocessor.py # Data preprocessing utilities
│   └── utils/             # Utility functions
└── report/                # Documentation and reports
```

## Data Directory Structure

The `data/` directory should be organized as follows:

```
data/
├── core/                    # Core survey data from each country
│   ├── ELSA/               # UK ELSA survey data
│   ├── HRS/                # US HRS survey data
│   ├── LASI/               # India LASI survey data
│   └── MHAS/               # Mexico MHAS survey data
├── hcap/                   # Harmonized Cognitive Assessment Protocol data
│   ├── ELSA/               # UK HCAP data
│   ├── HRS/                # US HCAP data
│   ├── LASI/               # India HCAP data
│   └── Mex-Cog/            # Mexico HCAP data
├── predictions/            # Model predictions output
├── processed/              # Processed and combined datasets
└── variables/              # Variable definitions and mappings
    ├── table_variables_type.xlsx     # Variable type specifications
    └── table_variables.xlsx          # Variable definitions and wave mappings
```

## Key Components

### 1. Data Preprocessing (`src/preprocess/`)

The `DataPreprocessor` class in `data_preprocessor.py` handles:
- Feature type handling (categorical, ordinal, numerical)
- Log transformations of numerical features
- Group-level aggregations
- One-hot encoding of categorical variables
- Missing value handling
- Feature cleaning and standardization

### 2. Model Implementations (`src/models/`)

The project implements several machine learning models:

- **Base Model** (`base_model.py`): Abstract base class providing common functionality:
  - Cross-validation
  - SHAP value computation
  - Confidence interval estimation
  - Sample weight handling for fairness

- **Individual Models**:
  - `lgbm_model.py`: LightGBM implementation
  - `catb_model.py`: CatBoost implementation
  - `xgbm_model.py`: XGBoost implementation
  
- **Ensemble Model** (`ensb_model.py`):
  - Age-aware soft-max blending of base models
  - Weighted combination based on performance
  - Support for SHAP values and confidence intervals

### 3. Pipeline Runner (`src/pipeline/`)

The `PipelineRunner` class in `pipeline_runner.py` orchestrates the entire ML workflow:

1. Data loading and splitting
2. Preprocessing with `DataPreprocessor`
3. Individual model training
4. Ensemble model training
5. Prediction generation
6. Model and data persistence
