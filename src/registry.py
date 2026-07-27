import mlflow


def load_registered_model(
    model_uri: str,
):

    return mlflow.pyfunc.load_model(
        model_uri,
    )