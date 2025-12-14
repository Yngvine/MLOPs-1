"""
Script to query MLflow for the best REGISTERED model and serialize it to ONNX.
"""
from mlflow.tracking import MlflowClient
from mlflow import pytorch as mlflow_pytorch
import torch
import os
import shutil

def serialize_best_model():
    # Setup Client and Model Name
    client = MlflowClient()
    model_name = "OxfordPetsMobileNetV2"
    
    print(f"Searching for registered versions of model: '{model_name}'...")

    # Query Registered Models
    try:
        registered_models = client.search_model_versions(f"name='{model_name}'")
    except Exception as e:
        print(f"Error searching models: {e}")
        return

    if not registered_models:
        print(f"No registered models found for name '{model_name}'.")
        return

    print(f"Found {len(registered_models)} registered versions.")

    # 3. Compare models to select the best one
    best_run_id = None
    best_acc = -1.0
    best_version_num = None

    for model_version in registered_models:
        run_id = model_version.run_id
        version = model_version.version
        
        if not run_id:
            print(f"Skipping version {version}: No run_id found.")
            continue

        # Get run info to access metrics
        try:
            run = client.get_run(run_id)
            val_acc = run.data.metrics.get("val_accuracy", 0.0)
            
            print(f"Version {version} (Run {run_id}): val_accuracy = {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_run_id = run_id
                best_version_num = version
        except Exception as e:
            print(f"Could not fetch metrics for version {version}: {e}")

    if best_run_id is None:
        print("Could not determine best model.")
        return

    print(f"\nBest Model: Version {best_version_num} (Run {best_run_id}) with Accuracy: {best_acc:.4f}")

    # Load the best model
    model_uri = f"runs:/{best_run_id}/model"
    print(f"Loading model from {model_uri}...")
    
    model = mlflow_pytorch.load_model(model_uri)
    model = model.to("cpu")
    model.eval()

    # Download Class Labels
    print("Downloading class labels...")
    local_artifacts_path = "model"
    os.makedirs(local_artifacts_path, exist_ok=True)
    
    # The artifact is stored in a folder named 'class_labels' inside the run
    try:
        downloaded_path = client.download_artifacts(run_id=best_run_id, path="class_labels", dst_path=local_artifacts_path)
        
        # Locate the JSON file inside the downloaded folder
        json_file = None
        for root, dirs, files in os.walk(downloaded_path):
            for file in files:
                if file.endswith(".json"):
                    json_file = os.path.join(root, file)
                    break
        
        if json_file:
            # Move/Rename to the expected location for the API
            final_labels_path = os.path.join(local_artifacts_path, "class_labels.json")
            shutil.move(json_file, final_labels_path)
            print(f"Class labels saved to {final_labels_path}")
            
            # Cleanup the downloaded folder if it's different from destination
            if downloaded_path != local_artifacts_path:
                 shutil.rmtree(downloaded_path, ignore_errors=True)
        else:
            print("Warning: No JSON file found in downloaded artifacts.")

    except Exception as e:
        print(f"Error downloading artifacts: {e}")

    # Serialize to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = os.path.join(local_artifacts_path, "oxford_pets_mobilenetv2.onnx")
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=18, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("Model serialized successfully.")

if __name__ == "__main__":
    serialize_best_model()
