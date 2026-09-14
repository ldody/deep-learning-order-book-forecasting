# Deep Learning & Full Order Book Dynamics for Stock Price Forecasting

Research code for a deep-learning pipeline designed to study
short-horizon stock-price forecasting from **Full Order Book (FOB)**
market microstructure data and conventional **OHLCV** data.

The repository accompanies the scientific article:

> **Léo Dody --- "Forecasting stock prices: Deep Learning and Full Order
> Book Dynamics"**\
> *Finance*, Association Française de Finance, 2026.

**Published article:**
https://shs.cairn.info/journal-finance-2026-0-page-I55?lang=en&tab=resume\
**Full Order Book data source:**
https://www.euronext.com/en/products-services/nexthistory

> **Portfolio note:** this repository focuses on the implementation and
> research pipeline. Empirical results from the published article are
> intentionally not reproduced here. Access to the article may require a
> subscription or institutional access.

------------------------------------------------------------------------

## Project overview

The project implements an end-to-end research workflow for processing
high-frequency market microstructure data and training deep-learning
models for multi-step stock-price forecasting.

The pipeline covers:

1.  management and reconstruction of raw Euronext Full Order Book
    events;
2.  extraction and aggregation of order-book features;
3.  integration with OHLCV market data;
4.  construction of model-ready datasets;
5.  hyperparameter optimization with Optuna;
6.  training of CNN, LSTM and hybrid CNN-LSTM architectures;
7.  forecast evaluation;
8.  sensitivity analysis using Morris and Sobol methods;
9.  parallel execution on an HPC cluster through Slurm job arrays.

The code was developed as research code and preserves the environment
and execution conventions used for the original experiments.

------------------------------------------------------------------------

## Main technical skills demonstrated

-   **Python** for scientific computing and data engineering
-   **Pandas / NumPy** for high-frequency financial data processing
-   **TensorFlow / Keras** for deep-learning model development
-   **CNN, LSTM and CNN-LSTM** architectures for time-series forecasting
-   **Optuna** for hyperparameter optimization
-   **SALib** for Morris and Sobol sensitivity analysis
-   **Scikit-learn** for preprocessing, validation and evaluation
-   **Parquet / Fastparquet / PyArrow** for efficient storage of large
    datasets
-   **FileLock** for concurrent experiment management
-   **Slurm** for distributed/HPC execution
-   Financial-market **microstructure and Full Order Book
    reconstruction**

------------------------------------------------------------------------

## Data

### Full Order Book

The raw Full Order Book data used by the pipeline originate from
**Euronext NextHistory**.

The repository does **not** redistribute the raw Euronext data. Users
wishing to reproduce the data-processing pipeline must obtain the
corresponding data separately and comply with Euronext's licensing
conditions.

The preprocessing code expects order-event information including fields
such as:

-   ISIN;
-   event date and time;
-   order identifier;
-   event type;
-   order side;
-   order price and size;
-   order type;
-   time in force;
-   trade price and size.

The pipeline reconstructs and aggregates several dimensions of
order-book activity, including limit-order-book depth, filled orders,
cancellations and time-in-force information.

### OHLCV

OHLCV files are used alongside the order-book features. The original
research dataset used market data that are **not redistributed in this
repository**.

The feature-building code expects OHLCV data containing:

`Open`, `High`, `Low`, `Close`, `Volume`

with a `Local Time` timestamp column.

### Asset universe

`data/assets.csv` contains the asset identifiers used by the
feature-building pipeline, including the mapping required by the code
between market identifiers such as ISIN and RIC.

------------------------------------------------------------------------

## Repository structure

``` text
.
├── data/
│   ├── assets.csv
│   ├── raw/
│   │   └── FOB/
│   └── processed/
│       └── OHLCV/
│
├── src/
│   ├── data/
│   │   ├── FOBDBM.py
│   │   ├── FOB_DB.csv
│   │   └── prepro_FOB.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── model_main.py
│   │   ├── model_build.py
│   │   ├── model_preprocessing.py
│   │   └── clean_optuna_failed.py
│   │
│   ├── metrics/
│   │   └── metrics_main.py
│   │
│   ├── GSA/
│   │   └── GSA_main.py
│   │
│   └── utils/
│       ├── NormStand.py
│       └── base_log.py
│
├── slurm/
│   ├── FOB_DB_reinit.bash
│   ├── FOB_prepro.bash
│   ├── build_features.bash
│   ├── opti.bash
│   ├── forecast.bash
│   ├── metrics.bash
│   └── GSA.bash
│
├── results/
├── notebooks/
├── tests/
├── requirements.txt
└── README.md
```

Generated intermediate directories and experiment outputs are
created/used by the pipeline according to the original research
environment.

------------------------------------------------------------------------

## Pipeline

``` text
Euronext Full Order Book
          │
          ▼
   FOB database manager
      (FOBDBM.py)
          │
          ▼
 FOB event preprocessing
    (prepro_FOB.py)
          │
          ├── LOB: reconstructed order-book information
          ├── FO : fill-order information
          ├── CO : cancellation information
          └── TIF: time-in-force information
          │
          ▼
       OHLCV data
          │
          ▼
    Feature engineering
  (build_features.py)
          │
          ▼
  Model preprocessing
          │
          ├── CNN
          ├── LSTM
          └── CNN-LSTM
          │
          ▼
 Hyperparameter optimization
         Optuna
          │
          ▼
 Training & forecasting
          │
          ├── Forecast metrics
          └── Morris / Sobol sensitivity analysis
```

------------------------------------------------------------------------

## Environment

The original HPC scripts load **Python 3.10.4**.

Create the virtual environment **inside the project root**:

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependencies declared by the original project are:

``` text
pandas
numpy
filelock
tensorflow
scikit-learn
matplotlib
pyarrow
fastparquet
optuna
salib
```

### Important compatibility note

This repository preserves the original research code without refactoring
it.

Some scripts locate the project root by looking for either:

-   a directory named exactly `PhD_article_1`; or
-   a `.venv` directory at the project root.

Therefore, **changing the GitHub repository name does not require
changing the source code**, but the local checkout should retain the
directory name expected by the original scripts.

For example, if the GitHub repository is named
`deep-learning-order-book-forecasting`, clone it as:

``` bash
git clone <YOUR-GITHUB-REPOSITORY-URL> PhD_article_1
cd PhD_article_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This preserves the assumptions of the original code while allowing the
public GitHub repository to have a clearer portfolio-oriented name.

------------------------------------------------------------------------

## Expected data layout

Before executing the complete pipeline, the corresponding licensed/raw
datasets must be placed in the locations expected by the original code.

A typical working layout is:

``` text
PhD_article_1/
├── .venv/
├── data/
│   ├── assets.csv
│   ├── raw/
│   │   ├── FOB/
│   │   │   └── <Euronext FOB archives>
│   │   └── OHLCV/
│   │       └── <OHLCV CSV files>
│   └── processed/
└── ...
```

The raw market datasets themselves are not included in this public
repository.

------------------------------------------------------------------------

## Running the pipeline

### 1. Initialize the FOB processing database

The database manager scans the available FOB archives and initializes
the processing state:

``` bash
python src/data/FOBDBM.py --reinit True
```

The processing database is used to coordinate Full Order Book
preprocessing tasks.

### 2. Preprocess Full Order Book events

The main preprocessing script reconstructs and aggregates the different
order-event components:

``` bash
python src/data/prepro_FOB.py -sa True --job_id <JOB_ID>
```

The script supports processing for:

-   `LOB` --- limit-order-book reconstruction;
-   `FO` --- filled orders;
-   `CO` --- cancelled orders;
-   `TIF` --- time-in-force information.

The original experiments parallelized this stage through Slurm arrays.

### 3. Build model features

After FOB preprocessing and once OHLCV files are available:

``` bash
python src/features/build_features.py
```

For each asset, the script loads the OHLCV series and combines them with
the processed LOB, fill-order, cancellation and time-in-force features.

The resulting datasets are stored as compressed Parquet files.

### 4. Hyperparameter optimization

The model pipeline uses Optuna for hyperparameter search:

``` bash
python src/models/model_main.py -ba True --job_id <JOB_ID>
```

The experiment configuration determines the asset, input dataset and
neural-network architecture associated with a job.

### 5. Train models and generate forecasts

For array-based execution:

``` bash
python src/models/model_main.py -sa True --job_id <JOB_ID>
```

The implemented model families include:

-   Convolutional Neural Networks (**CNN**);
-   Long Short-Term Memory networks (**LSTM**);
-   hybrid **CNN-LSTM** architectures.

Model construction is implemented in `src/models/model_build.py`, while
dataset preparation is handled by `src/models/model_preprocessing.py`.

### 6. Compute forecast metrics

``` bash
python src/metrics/metrics_main.py
```

The evaluation pipeline computes prediction-error and directional
metrics from the generated forecasts.

### 7. Run sensitivity analysis

``` bash
python src/GSA/GSA_main.py
```

The sensitivity-analysis pipeline uses **Morris screening** and **Sobol
global sensitivity analysis** through SALib to investigate the
contribution of model inputs.

------------------------------------------------------------------------

## HPC / Slurm execution

The original experiments were designed to run on an HPC infrastructure
using Slurm.

Example submission sequence:

``` bash
cd slurm

sbatch FOB_DB_reinit.bash
sbatch FOB_prepro.bash
sbatch build_features.bash
sbatch opti.bash
sbatch forecast.bash
sbatch metrics.bash
sbatch GSA.bash
```

These scripts document the computational workflow used during
development, including job arrays for parallel preprocessing,
optimization, forecasting and sensitivity analysis.

> **Important:** the `.bash` files contain paths, module definitions,
> partitions and environment settings specific to the original HPC
> infrastructure. They are preserved as part of the research code and
> must be adapted before being used on another cluster.

------------------------------------------------------------------------

## Experiment-management files

The modeling and analysis code also relies on experiment-state CSV files
such as:

``` text
data/assets_DB.csv
data/assets_GSA.csv
```

These files are used by the original workflow to allocate and track
combinations of assets, datasets, models and processing states,
particularly when multiple jobs run concurrently.

They are **not included in this public repository snapshot**.
Consequently, the repository should be viewed as the original
implementation associated with the research project rather than as a
self-contained benchmark with redistributable data and experiment
artifacts.

------------------------------------------------------------------------

## Reproducibility scope

This repository makes the **implementation** of the research workflow
visible, but it is not intended to redistribute the complete research
environment.

Full reproduction requires, among other things:

-   access to the original/licensed market datasets;
-   the expected OHLCV files;
-   experiment-management files used by the training pipeline;
-   sufficient computational resources for model optimization and
    training;
-   adaptation of the Slurm configuration when running outside the
    original HPC environment.

No synthetic or substitute dataset is provided because this repository
preserves the original research implementation.

------------------------------------------------------------------------

## Scientific publication

This code accompanies:

**Dody, L. (2026). *Forecasting stock prices: Deep Learning and Full
Order Book Dynamics*. Finance, Association Française de Finance.**

Publication page:\
https://shs.cairn.info/journal-finance-2026-0-page-I55?lang=en&tab=resume

The publication provides the scientific motivation, methodology and
academic discussion associated with the project.

This README deliberately does **not** reproduce the empirical findings
or numerical results of the published article.

------------------------------------------------------------------------

## Data source

Full Order Book data used in the research were obtained from:

**Euronext --- NextHistory**\
https://www.euronext.com/en/products-services/nexthistory

Euronext data are subject to the provider's applicable access and
licensing terms. No raw Euronext Full Order Book data are distributed
through this repository.

------------------------------------------------------------------------

## Disclaimer

This repository is provided for **research, educational and portfolio
purposes**.

It does not constitute investment advice, a trading recommendation, or a
production trading system. Financial-market forecasting is inherently
uncertain, and historical or experimental model performance does not
guarantee future performance.

------------------------------------------------------------------------

## Author

**Léo Dody**

Research interests: quantitative finance, market microstructure, machine
learning, deep learning and explainable AI for financial markets.
