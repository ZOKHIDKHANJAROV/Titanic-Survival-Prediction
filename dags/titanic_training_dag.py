import os
import subprocess
import sys
from pathlib import Path

import pendulum
from airflow.sdk import dag, task


PROJECT_ROOT = Path(
    os.environ.get(
        "TITANIC_PROJECT_ROOT",
        Path(__file__).resolve().parents[1],
    )
)
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def run_project_module(module_name, *arguments):
    command = [
        sys.executable,
        "-m",
        module_name,
        *[str(argument) for argument in arguments],
    ]
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
        check=True,
    )


@dag(
    dag_id="titanic_model_training",
    description="Prepare data, tune candidates, and publish the best model.",
    schedule=None,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    max_active_tasks=3,
    default_args={"retries": 1},
    tags=["machine-learning", "titanic"],
)
def titanic_model_training():
    @task
    def prepare_dataset():
        run_project_module(
            "src.prepare_data",
            "--params",
            PARAMS_PATH,
        )
        return str(PROJECT_ROOT / "data" / "processed")

    @task
    def get_enabled_models(_prepared_data):
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.settings import load_params
        from src.train import enabled_models

        return enabled_models(load_params(PARAMS_PATH))

    @task
    def train_candidate(model_name):
        run_project_module(
            "src.train",
            "--model",
            model_name,
            "--params",
            PARAMS_PATH,
        )
        return model_name

    @task
    def select_best_model(_trained_models):
        run_project_module(
            "src.select_model",
            "--params",
            PARAMS_PATH,
        )
        return str(PROJECT_ROOT / "models" / "best_model.pkl")

    prepared_data = prepare_dataset()
    model_names = get_enabled_models(prepared_data)
    trained_models = train_candidate.expand(model_name=model_names)
    select_best_model(trained_models)


titanic_training_dag = titanic_model_training()
