# Titanic MLOps Demo

A minimal but complete MLOps pipeline built for teaching.
Uses the Titanic dataset and a Decision Tree to demonstrate:
DVC (data + pipeline versioning), MLflow (experiment tracking + model registry), and GitHub Actions (CI/CD).

---

## Student Lab Guide

Follow these steps in order. Each step builds on the previous one.

### Prerequisites

Make sure you have these installed before starting:
- [Git](https://git-scm.com/downloads)
- Python 3.9 or higher (`python --version` to check)
- A [GitHub account](https://github.com)
- A [Kaggle account](https://www.kaggle.com) (free — needed to download the dataset)

---

### Step 1: Create your personal branch on GitHub

1. Open the repo in your browser: [https://github.com/MLEPalawat/titanic-mlops](https://github.com/MLEPalawat/titanic-mlops)
2. Click the **`main`** branch dropdown (top-left, just above the file list)
3. In the text box, type your branch name — use the format: `student-yourname`
   - Example: `student-alice`, `student-bob`
4. Click **"Create branch: student-yourname from main"**

Your personal branch now exists on GitHub. All your experiment changes will go here.

---

### Step 2: Clone the repository to your local machine

Open a terminal (Command Prompt, PowerShell, or bash) and run:

```bash
git clone https://github.com/MLEPalawat/titanic-mlops.git
cd titanic-mlops
```

---

### Step 3: Switch to your branch

```bash
# Replace "student-yourname" with the branch name you created in Step 1
git checkout student-yourname
```

Verify you're on the right branch:
```bash
git branch
# Output should show:  * student-yourname
```

---

### Step 4: Create a virtual environment and install dependencies

**Mac / Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You should see `(venv)` at the start of your terminal prompt once the environment is active.

---

### Step 5: Get the Titanic dataset

The dataset is tracked by DVC (not stored directly in git). Download it from Kaggle:

1. Go to [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
2. Download `train.csv`
3. Rename it to `titanic.csv`
4. Place it at: `data/raw/titanic.csv`

> **Why this step?** In real MLOps, large data files are NOT stored in git. DVC tracks them separately. The `.dvc` pointer file is in git; the actual data lives elsewhere (a DVC remote, cloud storage, etc.).

---

### Step 6: Run the full pipeline for the first time

```bash
dvc repro
```

You should see all three stages execute:
```
Running stage 'preprocess' ...
Running stage 'train' ...
Running stage 'evaluate' ...
```

This runs: **preprocess → train → evaluate** in order.

---

### Step 7: See DVC caching in action

Run the exact same command again immediately:

```bash
dvc repro
```

**Expected output:**
```
Stage 'preprocess' didn't change, skipping
Stage 'train' didn't change, skipping
Stage 'evaluate' didn't change, skipping
Data and pipelines are up to date.
```

> **What just happened?** DVC hashed every input file and parameter. Nothing changed, so it skipped all stages and used cached results. This is how DVC saves time in real projects.

---

### Step 8: Open the MLflow UI

Open a **second terminal window** (keep your first one open), activate the venv, and run:

```bash
# Mac/Linux
source venv/bin/activate
mlflow ui

# Windows
venv\Scripts\activate
mlflow ui
```

Open your browser and go to: **http://localhost:5000**

You should see the `titanic-survival` experiment with one run already logged from Step 6.

> Keep this terminal running the whole session. MLflow UI updates live as you run more experiments.

---

### Step 9: Run Experiment 2 — change max_depth

Open `params.yaml` in any text editor and change:

```yaml
model:
  max_depth: 4    # change this to 6
```

Save the file, then run:

```bash
dvc repro
```

**Expected output:**
```
Stage 'preprocess' didn't change, skipping     ← data unchanged, skipped
Running stage 'train' ...                       ← param changed, re-runs
Running stage 'evaluate' ...                    ← downstream, re-runs
```

Now check the differences:

```bash
dvc params diff      # shows: max_depth changed 4 → 6
dvc metrics show     # shows accuracy for current + previous run
```

**Refresh MLflow UI** — you should see a second run. Compare the two runs side by side.

---

### Step 10: Run Experiment 3 — trigger the quality gate (criterion: entropy)

Back in `params.yaml`, change:

```yaml
model:
  criterion: entropy    # was: gini
```

Save and run:

```bash
dvc repro
```

**Expected output (quality gate FAILS this time):**
```
[evaluate] Quality Gate Check:
  Accuracy : 0.7654
  Threshold: 0.7700
  Result   : FAIL  (0.7654 < 0.7700)
ERROR: failed to reproduce 'evaluate': failed to run: python src/evaluate.py, exited with 1
```

> **This is intentional.** `entropy` at `max_depth=6` produces a weaker model that falls below the minimum accuracy threshold. DVC marks the pipeline as failed — the same way GitHub Actions CI would fail.

The MLflow UI will still show this run (the train stage logged it before evaluate caught the failure). Refresh and you'll see 3 runs — but this one would be blocked from reaching production.

**Revert criterion before continuing:**
```yaml
model:
  criterion: gini   # revert back
```

---

### Step 11: Run Experiment 4 — change min_samples_leaf

Make sure `criterion` is back to `gini` from Step 10, then change:

```yaml
model:
  min_samples_leaf: 2   # was: 5
```

```bash
dvc repro
```

This should be your **best accuracy yet** (around 0.7989). Quality gate passes.

You now have 4 runs in MLflow. Notice how each parameter change creates a new tracked experiment automatically — you never lose a result.

---

### Step 12: Register the best model

Look at your MLflow UI runs. Find the run with the highest accuracy. Copy its **Run ID** (visible in the run details page).

```bash
# Register the best-performing run as a named model version
python src/register_model.py --run-id <paste-run-id-here>

# Or register the most recent run (no flag needed):
python src/register_model.py
```

In the MLflow UI, click the **"Models"** tab. You should see `titanic-decision-tree` version 1 with aliases **staging** and **production** assigned.

> **Note:** mlflow 3.x replaced the old Staging/Production stage system with aliases. The script uses `set_registered_model_alias()` and you load the model with `@alias` syntax instead of `/StageName`.

---

### Step 13: Commit and push your experiment results

```bash
# Stage params, metrics, and the DVC lock file (records exact pipeline state)
git add params.yaml metrics.json dvc.lock

git commit -m "experiment: max_depth=6, min_samples_leaf=2 — accuracy 0.7989"

git push origin student-yourname
```

Go to GitHub and open your branch. You can see your commit and compare it with classmates' branches.

---

### Step 14: Trigger a CI failure (optional demo)

In `params.yaml`, remove `Sex` from the feature list:

```yaml
features:
  use_columns:
    - Pclass
    # - Sex      ← remove this line
    - Age
    - SibSp
    - Parch
    - Fare
    - Embarked
```

Push to GitHub:

```bash
git add params.yaml
git commit -m "demo: remove Sex feature to trigger CI failure"
git push origin student-yourname
```

Go to **GitHub → Actions tab** on your branch. The CI pipeline will run and fail because accuracy drops below `min_accuracy: 0.77` in `params.yaml`.

Restore `Sex` to the feature list, push again, and watch the CI pass (green checkmark).

---

### Recap: What you just demonstrated

| Tool | What you used it for |
|---|---|
| **Git branches** | Isolated your experiment work from the main codebase |
| **DVC pipeline** | Ran 3-stage preprocess → train → evaluate pipeline |
| **DVC caching** | Saw unchanged stages automatically skipped |
| **DVC params diff** | Compared parameters between runs |
| **MLflow tracking** | Every `dvc repro` automatically logged a new experiment run |
| **MLflow comparison** | Compared accuracy across all 4 experiments side by side |
| **MLflow registry** | Promoted a model version to Staging |
| **GitHub Actions** | Saw a quality gate fail when model performance degraded |

---

## Project structure

```
titanic-mlops/
├── data/
│   ├── raw/titanic.csv           # Place the Kaggle CSV here (DVC tracked)
│   └── processed/features.csv   # Auto-generated by preprocess.py
├── src/
│   ├── preprocess.py             # Stage 1: clean + feature engineer
│   ├── train.py                  # Stage 2: train DT + log to MLflow
│   ├── evaluate.py               # Stage 3: metrics + quality gate
│   └── register_model.py         # Phase 4: promote model to registry
├── models/                       # Trained model pickle (DVC tracked)
├── .github/workflows/ci.yml      # GitHub Actions CI pipeline
├── dvc.yaml                      # DVC pipeline definition (3 stages)
├── params.yaml                   # ALL hyperparams and config (single source of truth)
├── metrics.json                  # Auto-generated by evaluate.py
└── requirements.txt
```

---

## Setup (do this once)

**1. Clone the repo and create a virtual environment**
```bash
git clone https://github.com/YOUR_USERNAME/titanic-mlops.git
cd titanic-mlops

python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

**2. Download the Titanic dataset**

Download `train.csv` from https://www.kaggle.com/c/titanic/data
Rename it to `titanic.csv` and place it at `data/raw/titanic.csv`

**3. Initialize DVC**
```bash
dvc init
git add .dvc/ .dvcignore
git commit -m "init: dvc setup"
```

**4. Add data to DVC tracking**
```bash
dvc add data/raw/titanic.csv
git add data/raw/titanic.csv.dvc data/raw/.gitignore
git commit -m "data: add raw titanic dataset to DVC"
```

---

## Phase 2: Run the DVC pipeline

```bash
# Run the full pipeline: preprocess → train → evaluate
dvc repro

# Run again — notice all stages are SKIPPED (nothing changed)
dvc repro

# Now change max_depth in params.yaml from 4 to 6, then:
dvc repro
# Only train + evaluate re-run. Preprocess is SKIPPED (data didn't change).

# See what changed between runs
dvc params diff
dvc metrics show
```

---

## Phase 3: Open the MLflow UI

```bash
# In a new terminal (keep this running while you experiment)
mlflow ui

# Open in browser: http://localhost:5000
```

Run the pipeline multiple times with different `params.yaml` values.
Each run appears as a new row in the MLflow UI — compare them side by side.

---

## Phase 4: Register a model version

```bash
# After running the pipeline at least once:
python src/register_model.py

# Or specify a particular run by ID:
python src/register_model.py --run-id <run_id_from_mlflow_ui>
```

Then in the MLflow UI, go to the **Models** tab to see the registry.

**Load the production model by alias (no file paths needed):**
```python
import mlflow.pyfunc
model = mlflow.pyfunc.load_model("models:/titanic-decision-tree@production")
predictions = model.predict(X_new)
```

> **mlflow 3.x:** Stages (Staging/Production) were replaced with aliases. Use `@alias` syntax instead of `/StageName`.

---

## Phase 5: GitHub Actions CI

Push to `main` to trigger the CI pipeline automatically:
```bash
git add .
git commit -m "feat: update features"
git push origin main
```

Go to **GitHub → Actions tab** to watch the pipeline run.

**Demo a CI failure:**
```yaml
# In params.yaml, remove 'Sex' from features.use_columns, then push.
# The model accuracy will drop below min_accuracy → CI fails (red X).
# Add 'Sex' back and push again → CI passes (green checkmark).
```

---

## Key commands reference

| Command | What it does |
|---|---|
| `dvc repro` | Run the full pipeline (skips unchanged stages) |
| `dvc status` | Show which stages are out of date |
| `dvc params diff` | Compare params between current state and last commit |
| `dvc metrics show` | Show metrics.json for current and previous runs |
| `mlflow ui` | Open the experiment tracking UI |
| `python src/register_model.py` | Register best model to MLflow registry |

---

## What each file teaches

| File | MLOps concept demonstrated |
|---|---|
| `params.yaml` | Config-as-code: hyperparams are not hardcoded |
| `dvc.yaml` | Pipeline DAG: reproducible, stage-based execution |
| `src/train.py` | Experiment tracking: every run is logged, nothing is lost |
| `src/evaluate.py` | Quality gate: automated performance threshold |
| `src/register_model.py` | Model versioning: register → alias as staging/production |
| `.github/workflows/ci.yml` | CI/CD: automated retraining + gating on every push |
