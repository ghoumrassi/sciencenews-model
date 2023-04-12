import mlflow


def init_mlflow_experiment(config):
    mlflow.set_experiment(config['mlflow_experiment_name'])
    run = mlflow.start_run()
    return run
