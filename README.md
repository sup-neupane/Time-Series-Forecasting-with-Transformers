# Time Series Forecasting with Transformers

Fine-tuning and training IBM Granite PatchTST for electricity load forecasting using the `granite-tsfm` library.

---

## Project Structure

```
Time-Series-Forecasting-with-Transformers/
├── load_data.csv                          # Electricity load dataset
├── Time-Series-Forecasting-with-Transformers.ipynb  # Main notebook
├── requirements.txt                       # Python dependencies
├── checkpoint/                            # Saved model checkpoints
│   └── patchtst/direct/train/
│       ├── output/                        # Model weights
│       └── logs/                          # Training logs
└── README.md
```

---

## Setup

### Requirements
- Python 3.11 (3.14 is not supported — ML packages lack pre-built wheels)
- CUDA 12.6 compatible GPU (tested on Windows)

### Installation

```bash
# 1. Create virtual environment with Python 3.11
py -3.11 -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# or
venv\Scripts\activate          # Windows (CMD)

# 2. Install PyTorch with CUDA support (must be installed first, separately)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install remaining dependencies
pip install -r requirements.txt
```

> **Windows note:** Set `NUM_WORKERS = 0` in the config — multiprocessing DataLoader workers crash on Windows with values > 0.

---

## Dataset

The dataset (`load_data.csv`) contains hourly electricity load readings with the following columns:

| Column | Description |
|---|---|
| `timestamp_utc` | UTC timestamp |
| `load_MW` | Electricity load in megawatts |

---

## Configuration

Key parameters at the top of the notebook:

```python
CSV_PATH         = 'load_data.csv'
TIMESTAMP_COLUMN = 'timestamp_utc'
TARGET_COLUMNS   = ['load_MW']
ID_COLUMNS       = []             # single time series
CONTEXT_LENGTH   = 168            # one week of hourly data
FORECAST_HORIZON = 24             # predict 24 hours ahead
PATCH_LENGTH     = 12
BATCH_SIZE       = 32
NUM_WORKERS      = 0              # keep 0 on Windows
TRAIN_FRAC       = 0.8
VALID_FRAC       = 0.1
```

---

## Model

### Option A — Train from Scratch (default)
A `PatchTSTForPrediction` model is built with a custom config:

```
context_length   : 168
patch_length     : 12
patch_stride     : 12
d_model          : 128
num_heads        : 16
num_layers       : 3
ffn_dim          : 512
dropout          : 0.2
norm             : batchnorm
```

### Option B — Fine-tune Pretrained IBM Granite Model
Uses `ibm-granite/granite-timeseries-patchtst` from Hugging Face.
Requires changing config to `CONTEXT_LENGTH=512`, `FORECAST_HORIZON=96`.

```python
model = PatchTSTForPrediction.from_pretrained(
    "ibm-granite/granite-timeseries-patchtst",
    config=config
)
```

---

## Training

```python
TrainingArguments(
    num_train_epochs        = 20,
    learning_rate           = 1e-4,
    per_device_train_batch  = 32,
    eval_strategy           = "epoch",
    load_best_model_at_end  = True,
    metric_for_best_model   = "eval_loss",
)

EarlyStoppingCallback(
    early_stopping_patience   = 5,
    early_stopping_threshold  = 0.001,
)
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error (MW) |
| RMSE | Root Mean Squared Error (MW) |
| MAPE | Mean Absolute Percentage Error (%) |

Results are inverse-scaled back to original MW units before metric computation.

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| Python 3.14 — numpy/granite-tsfm build failures | Use Python 3.11 (`py -3.11 -m venv venv`) |
| `torch.cuda` AssertionError — CUDA not enabled | Install torch via `--index-url https://download.pytorch.org/whl/cu126` separately before other packages |
| `ModuleNotFoundError: numpy.random.mtrand` | VS Code kernel pointing to wrong Python — select `Python 3.11 venv` interpreter |
| `PreTrainedModel` ImportError | `transformers` too new — pin to `4.45.0` |
| `ValueError: Input sequence length doesn't match` | `CONTEXT_LENGTH` mismatch between dataset and model config — must be identical |
| `KeyError: load_Mw` | Column name is case-sensitive — use exact name from `df.columns.tolist()` |
| `symlinks warning` on HuggingFace cache | Enable Windows Developer Mode in Settings → System → For Developers |
| `NUM_WORKERS > 0` crash on Windows | Set `NUM_WORKERS = 0` |

---

## Dependencies

Key packages and tested versions:

```
torch==2.10.0+cu126
torchvision==0.25.0+cu126
transformers==4.45.0
granite-tsfm==0.3.6
numpy==1.26.4
pandas>=2.0
scikit-learn
accelerate==0.34.0
peft==0.13.0
datasets
matplotlib
```

Generate a full pinned lockfile with:
```bash
pip freeze > requirements.txt
```

---

## License

Apache 2.0 — see [granite-tsfm](https://github.com/ibm-granite/granite-tsfm) and [Hugging Face Transformers](https://github.com/huggingface/transformers) for upstream licenses.