import mlflow.pyfunc

def load_production_model(model_name):
    """
    Charge le modèle actuellement en production depuis la MLflow Model Registry.
    """
    try:
        logged_model = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(logged_model)
        return model
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du modèle : {e}")
