"""
Script to query MLflow for the best model and serialize it to ONNX.
"""
import mlflow
from mlflow import pytorch as mlflow_pytorch
import torch
import os
import sys
import pandas as pd
from typing import cast

def serialize_best_model():
    experiment_name = "OxfordPets_MobileNetV2"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return

    # Search runs to find the best model based on validation accuracy
    runs = cast(pd.DataFrame,mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_accuracy DESC"],
        max_results=1
    ))

    if runs.empty:
        print("No runs found.")
        return

    best_run = runs.iloc[0]
    run_id = best_run.run_id
    best_acc = best_run["metrics.val_accuracy"]
    print(f"Best run ID: {run_id} with val_accuracy: {best_acc}")

    # Load the model from the best run
    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from {model_uri}...")
    model = mlflow_pytorch.load_model(model_uri)
    model = model.to("cpu")
    model.eval()

    # Create dummy input for ONNX export
    # MobileNetV2 expects 224x224 input
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export to ONNX
    output_path = "model/oxford_pets_mobilenetv2.onnx"
    print(f"Exporting model to {output_path}...")
    
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print("Model serialized successfully.")

if __name__ == "__main__":
    serialize_best_model()
