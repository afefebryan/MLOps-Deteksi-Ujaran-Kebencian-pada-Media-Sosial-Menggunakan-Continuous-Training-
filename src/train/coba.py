from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="http://localhost:5000")
client.search_model_versions("name='HateSpeechClassifierV1'")



versions = client.get_latest_versions("HateSpeechClassifierV1")
for v in versions:
    print(v.version, v.current_stage, v.source)