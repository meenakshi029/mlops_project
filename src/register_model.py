# =============================================================================
# src/register_model.py - Phase 4: MLflow Model Registry
# =============================================================================
# WHAT THIS SCRIPT DOES:
#   Takes the best run from MLflow and registers it in the Model Registry.
#   Demonstrates the full model lifecycle using aliases:
#     registered -> staging -> production
#
# NOTE: mlflow 3.x replaced stage names (Staging, Production) with aliases.
#   Old: client.transition_model_version_stage(name, version, stage="Production")
#   New: client.set_registered_model_alias(name, alias="production", version=...)
#   Load: mlflow.pyfunc.load_model("models:/model-name@production")
#
# WHEN TO RUN THIS:
#   After running train.py at least once and verifying the model is good.
#   This is intentionally NOT part of the automated DVC pipeline -
#   promotion to production is a deliberate human decision.
#
# HOW TO RUN:
#   python src/register_model.py
#
#   Optional: specify a run ID directly
#   python src/register_model.py --run-id <run_id_from_mlflow_ui>
#
# KEY TEACHING POINT:
#   After running this, show students how to LOAD a model by alias:
#     import mlflow.pyfunc
#     model = mlflow.pyfunc.load_model("models:/titanic-decision-tree@production")
#   This is how real production systems work - no file paths, just registry names.
# =============================================================================

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import yaml
import argparse
import sys

# ---------------------------------------------------------------------------
# Load parameters
# ---------------------------------------------------------------------------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

mlflow_params = params["mlflow"]
eval_params   = params["evaluation"]

TRACKING_URI  = mlflow_params["tracking_uri"]
EXPERIMENT    = mlflow_params["experiment_name"]
MODEL_NAME    = mlflow_params["model_name"]
MIN_ACCURACY  = eval_params["min_accuracy"]


def get_best_run(client: MlflowClient, experiment_name: str) -> str:
    """
    Find the best MLflow run by accuracy metric.

    Searches all runs in the experiment and returns the run_id of the
    run with the highest test accuracy.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"[register] ERROR: Experiment '{experiment_name}' not found.")
        print("  -> Run `python src/train.py` first to create runs.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )

    if not runs:
        print("[register] ERROR: No runs found in the experiment.")
        sys.exit(1)

    best_run = runs[0]
    accuracy  = best_run.data.metrics.get("accuracy", 0)

    print(f"[register] Best run found:")
    print(f"  Run ID  : {best_run.info.run_id}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Params  : max_depth={best_run.data.params.get('max_depth')}, "
          f"criterion={best_run.data.params.get('criterion')}")

    return best_run.info.run_id, accuracy


def register_model(client: MlflowClient, run_id: str, model_name: str) -> str:
    """
    Register a model version from an existing MLflow run.

    Creates a new version under the given model name.
    """
    model_uri = f"runs:/{run_id}/decision-tree-model"

    print(f"\n[register] Registering model...")
    print(f"  Model URI: {model_uri}")
    print(f"  Name     : {model_name}")

    model_version = mlflow.register_model(
        model_uri  = model_uri,
        name       = model_name,
    )

    version = model_version.version
    print(f"  Version  : {version}")

    return version


def promote_to_staging(client: MlflowClient, model_name: str, version: str) -> None:
    """
    Tag a model version with alias 'staging'.

    In mlflow 3.x, stages were replaced with aliases.
    'staging' = ready for QA / validation, not yet in production.
    """
    client.set_registered_model_alias(
        name    = model_name,
        alias   = "staging",
        version = version,
    )
    print(f"\n[register] Version {version} aliased as: staging")


def promote_to_production(client: MlflowClient, model_name: str, version: str,
                           accuracy: float) -> None:
    """
    Tag a model version with alias 'production' if it passes the quality gate.

    TEACHING POINT:
      In real systems, this promotion would be triggered by CI after
      a successful canary deploy evaluation, not just accuracy on test set.
    """
    if accuracy < MIN_ACCURACY:
        print(f"\n[register] BLOCKED: accuracy {accuracy:.4f} < threshold {MIN_ACCURACY}")
        print("  -> Model stays in staging. Fix the model before promoting to production.")
        return

    client.set_registered_model_alias(
        name    = model_name,
        alias   = "production",
        version = version,
    )
    print(f"\n[register] Version {version} aliased as: production")
    print(f"\n  Load this model anywhere with:")
    print(f"    import mlflow.pyfunc")
    print(f"    model = mlflow.pyfunc.load_model('models:/{model_name}@production')")


def demo_load_from_registry(model_name: str) -> None:
    """
    Demonstrate loading a model by alias from the registry.

    This is the 'aha moment' - no file paths, no pickle files.
    Just ask the registry for the production model by name.
    """
    print(f"\n[register] Demo: Loading production model from registry...")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
        print(f"  Model loaded successfully!")
        print(f"  Type: {type(model)}")
        print(f"  -> In production code you would call: model.predict(X_new)")
    except Exception as e:
        print(f"  Could not load model: {e}")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register MLflow model to registry")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Specific MLflow run ID to register (default: best run by accuracy)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4: MLflow Model Registry")
    print("=" * 60)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    if args.run_id:
        run      = client.get_run(args.run_id)
        run_id   = args.run_id
        accuracy = run.data.metrics.get("accuracy", 0)
        print(f"[register] Using specified run: {run_id} (accuracy: {accuracy:.4f})")
    else:
        run_id, accuracy = get_best_run(client, EXPERIMENT)

    version = register_model(client, run_id, MODEL_NAME)
    promote_to_staging(client, MODEL_NAME, version)
    promote_to_production(client, MODEL_NAME, version, accuracy)
    demo_load_from_registry(MODEL_NAME)

    print("\n[register] Done!")
    print(f"[register] Open MLflow UI to see the registry:")
    print(f"  mlflow ui -> http://localhost:5000 -> Models tab")
