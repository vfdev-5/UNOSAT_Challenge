
def get_artifact_path(run_uuid, path):
    import mlflow
    client = mlflow.tracking.MlflowClient()
    return client.download_artifacts(run_uuid, path)
