#!/usr/bin/env python3
"""
Multi-Model Age Classification App - Cleaned Version
Enhanced version supporting multiple age detection models with harmonized outputs
Removed misleading FLIP implementation - now focuses on proven models
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification, SiglipForImageClassification
from PIL import Image
import gradio as gr
import numpy as np
import glob
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelAgeClassifier:
    def __init__(self):
        self.models = {
            "nateraw": {
                "name": "nateraw/vit-age-classifier",
                "model": None,
                "processor": None,
                "architecture": "vit",
                "age_labels": ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"],
                "underage_indices": [0, 1, 2],  # 0-2, 3-9, 10-19
                "loaded": False
            },
            "prithiv": {
                "name": "prithivMLmods/Age-Classification-SigLIP2",
                "model": None,
                "processor": None,
                "architecture": "siglip",
                "age_labels": ["Child 0-12", "Teenager 13-20", "Adult 21-44", "Middle Age 45-64", "Aged 65+"],
                "underage_indices": [0, 1],  # Child 0-12, Teenager 13-20
                "loaded": False
            }
        }

        # Harmonized age groups for comparison
        self.harmonized_groups = {
            "Child (0-12)": {"nateraw": [0, 1], "prithiv": [0]},           # 0-2, 3-9 -> Child 0-12
            "Teenager (13-19)": {"nateraw": [2], "prithiv": [1]},          # 10-19 -> Teenager 13-20
            "Young Adult (20-29)": {"nateraw": [3], "prithiv": [2]},       # 20-29 -> Adult 21-44 (partial)
            "Adult (30-44)": {"nateraw": [4], "prithiv": [2]},             # 30-39 -> Adult 21-44 (partial)
            "Middle Age (45-64)": {"nateraw": [5, 6], "prithiv": [3]},     # 40-49, 50-59 -> Middle Age 45-64
            "Senior (65+)": {"nateraw": [7, 8], "prithiv": [4]}            # 60-69, 70+ -> Aged 65+
        }

        self.active_models = ["nateraw"]  # Default to original model

    def load_model(self, model_key):
        """Load a specific model"""
        if model_key not in self.models:
            return False

        model_info = self.models[model_key]

        try:
            print(f"Loading {model_info['name']}...")

            # Load processor (same for both models)
            model_info["processor"] = AutoImageProcessor.from_pretrained(model_info["name"])

            # Load model based on architecture
            if model_info["architecture"] == "vit":
                model_info["model"] = AutoModelForImageClassification.from_pretrained(model_info["name"])
            elif model_info["architecture"] == "siglip":
                model_info["model"] = SiglipForImageClassification.from_pretrained(model_info["name"])

            model_info["loaded"] = True
            print(f"{model_info['name']} loaded successfully!")
            return True

        except Exception as e:
            print(f"Error loading {model_info['name']}: {e}")
            model_info["loaded"] = False
            return False

    def load_active_models(self):
        """Load all active models"""
        success_count = 0
        for model_key in self.active_models:
            if self.load_model(model_key):
                success_count += 1
        return success_count > 0

    def classify_age_single_model(self, image, model_key):
        """Classify age using a single model"""
        if model_key not in self.models or not self.models[model_key]["loaded"]:
            return {"error": f"Model {model_key} not loaded"}

        model_info = self.models[model_key]

        try:
            # Standard model processing
            inputs = model_info["processor"](image, return_tensors="pt")

            # Get prediction
            with torch.no_grad():
                outputs = model_info["model"](**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Get probabilities for each age group
            probabilities = predictions[0].numpy()

            # Create results dictionary
            results = {}
            for i, label in enumerate(model_info["age_labels"]):
                results[label] = float(probabilities[i])

            # Get predicted age group
            predicted_idx = np.argmax(probabilities)
            predicted_age = model_info["age_labels"][predicted_idx]
            confidence = float(probabilities[predicted_idx])

            return {
                "model": model_key,
                "model_name": model_info["name"],
                "predicted_age": predicted_age,
                "confidence": confidence,
                "all_probabilities": results,
                "raw_probabilities": probabilities
            }

        except Exception as e:
            return {"error": f"Classification error for {model_key}: {str(e)}"}

    def harmonize_probabilities(self, model_results):
        """Convert model-specific probabilities to harmonized age groups"""
        harmonized = {}

        for model_key, result in model_results.items():
            if "error" in result:
                continue

            # Standard processing for nateraw and prithiv
            raw_probs = result["raw_probabilities"]
            harmonized[model_key] = {}

            for harmonized_group, mapping in self.harmonized_groups.items():
                if model_key in mapping:
                    # Sum probabilities for indices that map to this harmonized group
                    prob_sum = sum(raw_probs[idx] for idx in mapping[model_key])
                    harmonized[model_key][harmonized_group] = prob_sum

        return harmonized

    def calculate_ensemble_prediction(self, harmonized_results, method="average"):
        """Calculate ensemble prediction from harmonized results"""
        if not harmonized_results:
            return {"error": "No valid model results for ensemble"}

        # Get all harmonized groups
        groups = list(self.harmonized_groups.keys())
        ensemble_probs = {}

        for group in groups:
            if method == "average":
                # Simple average of available models
                probs = [harmonized_results[model][group]
                        for model in harmonized_results
                        if group in harmonized_results[model]]
                ensemble_probs[group] = np.mean(probs) if probs else 0.0

            elif method == "weighted":
                # Could add model-specific weights here
                probs = [harmonized_results[model][group]
                        for model in harmonized_results
                        if group in harmonized_results[model]]
                ensemble_probs[group] = np.mean(probs) if probs else 0.0

        # Get ensemble prediction
        predicted_group = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[predicted_group]

        return {
            "predicted_age": predicted_group,
            "confidence": confidence,
            "all_probabilities": ensemble_probs,
            "method": method
        }

    def analyze_model_agreement(self, model_results, harmonized_results):
        """Analyze agreement between models"""
        if len(model_results) < 2:
            return {"error": "Need at least 2 models for agreement analysis"}

        valid_results = {k: v for k, v in model_results.items() if "error" not in v}

        if len(valid_results) < 2:
            return {"error": "Need at least 2 valid model results"}

        # Check prediction agreement
        predictions = [result["predicted_age"] for result in valid_results.values()]
        unique_predictions = set(predictions)

        agreement_stats = {
            "total_models": len(valid_results),
            "unique_predictions": len(unique_predictions),
            "full_agreement": len(unique_predictions) == 1,
            "predictions_by_model": {model: result["predicted_age"]
                                   for model, result in valid_results.items()},
            "confidence_by_model": {model: result["confidence"]
                                  for model, result in valid_results.items()}
        }

        # Calculate harmonized agreement
        if harmonized_results:
            harmonized_predictions = {}
            for model_key in harmonized_results:
                max_group = max(harmonized_results[model_key],
                              key=harmonized_results[model_key].get)
                harmonized_predictions[model_key] = max_group

            unique_harmonized = set(harmonized_predictions.values())
            agreement_stats["harmonized_agreement"] = len(unique_harmonized) == 1
            agreement_stats["harmonized_predictions"] = harmonized_predictions

        # Calculate confidence spread
        confidences = list(agreement_stats["confidence_by_model"].values())
        agreement_stats["confidence_std"] = np.std(confidences)
        agreement_stats["confidence_range"] = max(confidences) - min(confidences)

        # Agreement quality assessment
        if agreement_stats["full_agreement"] and agreement_stats["confidence_std"] < 0.1:
            agreement_stats["agreement_quality"] = "Strong Agreement"
        elif agreement_stats.get("harmonized_agreement", False):
            agreement_stats["agreement_quality"] = "Harmonized Agreement"
        elif agreement_stats["confidence_range"] < 0.2:
            agreement_stats["agreement_quality"] = "Similar Confidence"
        else:
            agreement_stats["agreement_quality"] = "Disagreement"

        return agreement_stats

    def classify_age_multi_model(self, image, include_ensemble=True, ensemble_method="average"):
        """Classify age using multiple models with ensemble and agreement analysis"""
        if image is None:
            return {"error": "Input image is missing"}
            
        if not self.active_models:
            return {"error": "No active models selected"}

        # Classify with each active model
        model_results = {}
        for model_key in self.active_models:
            result = self.classify_age_single_model(image, model_key)
            model_results[model_key] = result

        # Check if any models succeeded
        valid_results = {k: v for k, v in model_results.items() if "error" not in v}
        if not valid_results:
            return {"error": "All models failed to classify image"}

        # Harmonize probabilities
        harmonized_results = self.harmonize_probabilities(valid_results)

        # Calculate ensemble prediction
        ensemble_result = None
        if include_ensemble and len(valid_results) > 1:
            ensemble_result = self.calculate_ensemble_prediction(harmonized_results, ensemble_method)

        # Analyze model agreement
        agreement_analysis = None
        if len(valid_results) > 1:
            agreement_analysis = self.analyze_model_agreement(valid_results, harmonized_results)

        return {
            "individual_results": model_results,
            "harmonized_results": harmonized_results,
            "ensemble_result": ensemble_result,
            "agreement_analysis": agreement_analysis,
            "active_models": self.active_models,
            "successful_models": list(valid_results.keys())
        }

    def is_underage_multi_model(self, image, threshold=0.5, adjustment_mode="balanced"):
        """Check if person appears underage using multiple models with adjustable correction levels"""
        if image is None:
            return {"error": "Input image is missing"}

        result = self.classify_age_multi_model(image, include_ensemble=True)

        if "error" in result:
            return result

        underage_results = {}

        # Define adjustment factors based on mode
        adjustment_factors = {
            "strict": {"nateraw": 1.0, "prithiv": 1.0},      # No adjustment (strict original outputs)
            "balanced": {"nateraw": 0.9, "prithiv": 0.8}     # Moderate adjustment
        }

        factors = adjustment_factors.get(adjustment_mode, adjustment_factors["strict"])

        # Check each individual model with adjustable correction
        for model_key, model_result in result.get("individual_results", {}).items():
            if "error" not in model_result:
                model_info = self.models[model_key]
                
                if adjustment_mode != "strict":
                    # Model-specific underage calculation with adjustments
                    raw_probs = model_result["raw_probabilities"]
                    
                    if model_key == "nateraw":
                        # 0-2, 3-9: Full probability (definitely underage)
                        # 10-19: Adjusted probability (some adults in this range)
                        young_prob = raw_probs[0] + raw_probs[1]  # 0-2, 3-9
                        teen_prob = raw_probs[2]  # 10-19

                        adjusted_teen_prob = teen_prob * factors["nateraw"]
                        underage_prob = young_prob + adjusted_teen_prob

                    elif model_key == "prithiv":
                        # Child 0-12: Full probability (definitely underage)
                        # Teenager 13-20: Adjusted probability (some adults in this range)
                        child_prob = raw_probs[0]  # Child 0-12
                        teen_prob = raw_probs[1]   # Teenager 13-20

                        adjusted_teen_prob = teen_prob * factors["prithiv"]
                        underage_prob = child_prob + adjusted_teen_prob

                    else:
                        # Default fallback
                        underage_indices = model_info["underage_indices"]
                        underage_prob = sum(raw_probs[i] for i in underage_indices)
                else:
                    # Strict approach (no adjustment)
                    raw_probs = model_result["raw_probabilities"]
                    underage_indices = model_info["underage_indices"]
                    underage_prob = sum(raw_probs[i] for i in underage_indices)

                is_minor = underage_prob > threshold

                underage_results[model_key] = {
                    "is_underage": is_minor,
                    "underage_probability": underage_prob,
                    "threshold": threshold,
                    "adjustment_mode": adjustment_mode,
                    "adjustment_factor": factors.get(model_key, 1.0) if adjustment_mode != "strict" else 1.0
                }

        # Ensemble underage decision with adjustments
        ensemble_underage = None
        if result.get("ensemble_result"):
            if adjustment_mode != "strict":
                # Balanced ensemble calculation
                child_prob = result["ensemble_result"]["all_probabilities"].get("Child (0-12)", 0)
                teen_prob = result["ensemble_result"]["all_probabilities"].get("Teenager (13-19)", 0)

                # Apply moderate correction for ensemble
                ensemble_factor = 0.85  # Moderate for ensemble
                adjusted_teen_prob = teen_prob * ensemble_factor
                ensemble_underage_prob = child_prob + adjusted_teen_prob
            else:
                # Strict approach
                underage_groups = ["Child (0-12)", "Teenager (13-19)"]
                ensemble_underage_prob = sum(
                    result["ensemble_result"]["all_probabilities"].get(group, 0)
                    for group in underage_groups
                )

            ensemble_underage = {
                "is_underage": ensemble_underage_prob > threshold,
                "underage_probability": ensemble_underage_prob,
                "threshold": threshold,
                "adjustment_mode": adjustment_mode
            }

        return {
            "individual_underage": underage_results,
            "ensemble_underage": ensemble_underage,
            "agreement_analysis": result.get("agreement_analysis"),
            "threshold": threshold,
            "adjustment_mode": adjustment_mode
        }

    def set_active_models(self, model_list):
        """Set which models to use for classification"""
        self.active_models = [model for model in model_list if model in self.models]
        return self.load_active_models()

    def get_available_models(self):
        """Get list of available models with their status"""
        return {
            model_key: {
                "name": info["name"],
                "loaded": info["loaded"],
                "architecture": info["architecture"],
                "age_groups": len(info["age_labels"])
            }
            for model_key, info in self.models.items()
        }

    def analyze_distribution(self, probabilities):
        """Analyze age probability distribution with multiple statistical measures"""
        probs = np.array(list(probabilities.values()))

        # 1. Entropy (lower = more confident/concentrated)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy

        # 2. Gini Coefficient (higher = more concentrated)
        sorted_probs = np.sort(probs)
        n = len(probs)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_probs))) / (n * np.sum(sorted_probs)) - (n + 1) / n

        # 3. Peak Ratio (how much the highest probability dominates)
        max_prob = np.max(probs)
        avg_prob = np.mean(probs)
        peak_ratio = max_prob / avg_prob

        # 4. Standard Deviation (spread of probabilities)
        std_dev = np.std(probs)

        # 5. Distribution Type Classification
        if normalized_entropy < 0.3:
            dist_type = "Very Confident"
        elif normalized_entropy < 0.5:
            dist_type = "Confident"
        elif normalized_entropy < 0.7:
            dist_type = "Moderate"
        elif normalized_entropy < 0.85:
            dist_type = "Uncertain"
        else:
            dist_type = "Very Uncertain/Flat"

        return {
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "gini_coefficient": gini,
            "peak_ratio": peak_ratio,
            "standard_deviation": std_dev,
            "distribution_type": dist_type
        }

    def batch_process_folder(self, folder_path, underage_threshold=0.5, adjustment_mode="balanced", progress_callback=None):
        """Process all images in a folder with multi-model analysis"""
        if not folder_path or not os.path.exists(folder_path):
            return {"error": f"Folder not found: {folder_path}"}

        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.webp']

        # Find all image files
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(folder_path, '**', ext)
            image_files.extend(glob.glob(pattern, recursive=True))
            # Also check uppercase
            pattern = os.path.join(folder_path, '**', ext.upper())
            image_files.extend(glob.glob(pattern, recursive=True))

        # Remove duplicates (Windows glob is case-insensitive, so we get duplicates)
        image_files = list(set(image_files))

        if not image_files:
            return {"error": "No image files found in folder"}

        results = []
        total_files = len(image_files)

        for i, image_path in enumerate(image_files):
            try:
                # Update progress
                if progress_callback:
                    progress_callback(i / total_files, f"Processing {os.path.basename(image_path)}...")

                # Load and classify image
                image = Image.open(image_path).convert('RGB')

                # Multi-model classification
                multi_result = self.classify_age_multi_model(image, include_ensemble=True)
                underage_result = self.is_underage_multi_model(image, underage_threshold, adjustment_mode)

                if "error" not in multi_result and "error" not in underage_result:
                    # Extract results for each model
                    individual_results = {}
                    for model_key in self.active_models:
                        if model_key in multi_result.get("individual_results", {}):
                            model_result = multi_result["individual_results"][model_key]
                            if "error" not in model_result:
                                # Add distribution analysis
                                dist_analysis = self.analyze_distribution(model_result["all_probabilities"])
                                individual_results[model_key] = {
                                    "predicted_age": model_result["predicted_age"],
                                    "confidence": model_result["confidence"],
                                    "distribution_analysis": dist_analysis,
                                    "all_probabilities": model_result["all_probabilities"]
                                }

                    # Ensemble results
                    ensemble_result = multi_result.get("ensemble_result")
                    ensemble_underage = underage_result.get("ensemble_underage")

                    # Agreement analysis
                    agreement = multi_result.get("agreement_analysis", {})

                    result = {
                        "filename": os.path.basename(image_path),
                        "filepath": image_path,
                        "individual_results": individual_results,
                        "ensemble_result": ensemble_result,
                        "ensemble_underage": ensemble_underage,
                        "agreement_analysis": agreement,
                        "individual_underage": underage_result.get("individual_underage", {}),
                        "active_models": self.active_models,
                        "adjustment_mode": underage_result.get("adjustment_mode", "strict")
                    }
                else:
                    result = {
                        "filename": os.path.basename(image_path),
                        "filepath": image_path,
                        "error": multi_result.get("error", underage_result.get("error", "Unknown error"))
                    }

                results.append(result)

            except Exception as e:
                results.append({
                    "filename": os.path.basename(image_path),
                    "filepath": image_path,
                    "error": str(e)
                })

        if progress_callback:
            progress_callback(1.0, "Processing complete!")

        return {
            "total_files": total_files,
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results,
            "active_models": self.active_models,
            "adjustment_mode": adjustment_mode
        }

    def save_batch_results(self, batch_results, output_format="csv"):
        """Save multi-model batch results to file"""
        if "error" in batch_results:
            return {"error": batch_results["error"]}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            if output_format == "csv":
                filename = f"multi_model_age_results_{timestamp}.csv"

                # Prepare data for CSV
                csv_data = []
                for result in batch_results["results"]:
                    if "error" not in result:
                        row = {
                            "filename": result["filename"],
                            "filepath": result["filepath"]
                        }

                        # Add ensemble results
                        if result.get("ensemble_result"):
                            ensemble = result["ensemble_result"]
                            row["ensemble_predicted_age"] = ensemble["predicted_age"]
                            row["ensemble_confidence"] = ensemble["confidence"]
                            row["ensemble_method"] = ensemble["method"]

                        if result.get("ensemble_underage"):
                            row["ensemble_is_underage"] = result["ensemble_underage"]["is_underage"]
                            row["ensemble_underage_probability"] = result["ensemble_underage"]["underage_probability"]

                        # Add agreement analysis
                        if result.get("agreement_analysis"):
                            agreement = result["agreement_analysis"]
                            row["agreement_quality"] = agreement.get("agreement_quality", "Unknown")
                            row["full_agreement"] = agreement.get("full_agreement", False)
                            row["confidence_std"] = agreement.get("confidence_std", 0)

                        # Add individual model results
                        for model_key, model_result in result.get("individual_results", {}).items():
                            prefix = f"{model_key}_"
                            row[f"{prefix}predicted_age"] = model_result["predicted_age"]
                            row[f"{prefix}confidence"] = model_result["confidence"]
                            row[f"{prefix}distribution_type"] = model_result["distribution_analysis"]["distribution_type"]
                            row[f"{prefix}entropy"] = model_result["distribution_analysis"]["entropy"]
                            row[f"{prefix}peak_ratio"] = model_result["distribution_analysis"]["peak_ratio"]

                            # Add underage results for this model
                            if model_key in result.get("individual_underage", {}):
                                underage_info = result["individual_underage"][model_key]
                                row[f"{prefix}is_underage"] = underage_info["is_underage"]
                                row[f"{prefix}underage_probability"] = underage_info["underage_probability"]

                            # Add all age probabilities for this model
                            for age_group, prob in model_result["all_probabilities"].items():
                                safe_age_group = age_group.replace(" ", "_").replace("-", "_").replace("+", "plus")
                                row[f"{prefix}prob_{safe_age_group}"] = prob
                    else:
                        row = {
                            "filename": result["filename"],
                            "filepath": result["filepath"],
                            "error": result["error"]
                        }
                    csv_data.append(row)

                df = pd.DataFrame(csv_data)
                df.to_csv(filename, index=False)

            elif output_format == "json":
                filename = f"multi_model_age_results_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(batch_results, f, indent=2)

            return {"saved_file": filename, "total_results": len(batch_results["results"])}

        except Exception as e:
            return {"error": f"Failed to save file: {str(e)}"}


# Initialize multi-model classifier
multi_classifier = MultiModelAgeClassifier()

def extract_filename_from_path(file_path):
    """Extract filename from uploaded file path, handling None and various formats"""
    if file_path is None:
        return ""
    
    try:
        # Handle if it's already a filename or if it's a full path
        if isinstance(file_path, str):
            # Extract just the filename without extension for cleaner labels
            filename = os.path.basename(file_path)
            name_without_ext = os.path.splitext(filename)[0]
            return name_without_ext
        else:
            return ""
    except Exception:
        return ""

def update_label_from_upload(uploaded_file):
    """Update label when file is uploaded"""
    filename = extract_filename_from_path(uploaded_file)
    return filename if filename else "Image"

def batch_process_folder_multi(folder_path, selected_models, threshold, adjustment_mode, file_format, progress=gr.Progress()):
    """Gradio interface for multi-model batch processing"""
    if not folder_path or not os.path.exists(folder_path):
        return "Please select a valid folder path", None, "", None

    if not selected_models:
        return "Please select at least one model", None, "", None

    if not multi_classifier.set_active_models(selected_models):
        return "Failed to load selected models", None, "", None

    def progress_callback(completed, message):
        progress(completed, desc=message)

    batch_results = multi_classifier.batch_process_folder(
        folder_path,
        underage_threshold=threshold,
        adjustment_mode=adjustment_mode,
        progress_callback=progress_callback
    )

    if "error" in batch_results:
        return f"Error: {batch_results['error']}", None, "", None

    # --- Create Summary Table ---
    table_data = []
    for result in batch_results["results"]:
        if "error" not in result:
            row = {
                "filename": result.get("filename"),
                "ensemble_underage_probability": result.get("ensemble_underage", {}).get("underage_probability"),
                "nateraw_underage_probability": result.get("individual_underage", {}).get("nateraw", {}).get("underage_probability"),
                "nateraw_distribution_type": result.get("individual_results", {}).get("nateraw", {}).get("distribution_analysis", {}).get("distribution_type"),
                "prithiv_underage_probability": result.get("individual_underage", {}).get("prithiv", {}).get("underage_probability"),
                "prithiv_distribution_type": result.get("individual_results", {}).get("prithiv", {}).get("distribution_analysis", {}).get("distribution_type"),
            }
            table_data.append(row)

    if not table_data:
        return "No results to display.", None, "Completed with no valid results.", pd.DataFrame()

    df = pd.DataFrame(table_data)
    # Sort by ensemble probability in descending order
    if "ensemble_underage_probability" in df.columns and not df["ensemble_underage_probability"].isnull().all():
        df = df.sort_values(by="ensemble_underage_probability", ascending=False, na_position='last').reset_index(drop=True)

    # --- Styling ---
    def style_underage_probs(val, threshold_val):
        if pd.isna(val):
            return ''
        color = '#ffc7ce' if val > threshold_val else '#c6efce'
        font_color = '#9c0006' if val > threshold_val else '#006100'
        return f'background-color: {color}; color: {font_color};'

    prob_cols = [col for col in df.columns if "probability" in col]
    
    styled_df = df.style.apply(
        lambda s: s.map(lambda x: style_underage_probs(x, threshold_val=threshold)),
        subset=prob_cols
    ).format(
        {col: '{:.2%}' for col in prob_cols},
        na_rep=""
    )
    
    # --- Save full results ---
    save_result = multi_classifier.save_batch_results(batch_results, file_format)
    if "error" in save_result:
        status_msg = f"Error saving file: {save_result['error']}"
        saved_file = None
    else:
        status_msg = f"Results saved to {save_result['saved_file']}"
        saved_file = save_result['saved_file']

    summary_text = f"Processed {batch_results['total_files']} files. Found {batch_results['successful']} valid results."

    return summary_text, saved_file, status_msg, styled_df

def compare_two_images_multi_model(image1, image2, label1, label2, selected_models, adjustment_mode):
    """Compare two images with multi-model analysis, generating a separate large chart for each model."""
    if image1 is None or image2 is None:
        return None, None, "Please upload both images"

    if not selected_models:
        return None, None, "Please select at least one model"

    # Use fallback labels if the user-provided ones are empty
    if not label1:
        label1 = "Image 1"
    if not label2:
        label2 = "Image 2"

    # Set active models
    if not multi_classifier.set_active_models(selected_models):
        return None, None, "Failed to load selected models"

    # Analyze both images
    result1 = multi_classifier.classify_age_multi_model(image1, include_ensemble=True)
    result2 = multi_classifier.classify_age_multi_model(image2, include_ensemble=True)
    underage1 = multi_classifier.is_underage_multi_model(image1, 0.5, adjustment_mode)
    underage2 = multi_classifier.is_underage_multi_model(image2, 0.5, adjustment_mode)

    if "error" in result1 or "error" in result2:
        return None, None, f"Error analyzing images: {result1.get('error', '')} {result2.get('error', '')}"

    # --- Generate Comparison Summary ---
    summary = f"## 🆚 Multi-Model Comparison: {label1} vs {label2}\n\n"
    underage_summary = "### 🚨 Underage Probability Comparison\n\n"
    underage_summary += "| Model | " + f"{label1}" + " | " + f"{label2}" + " | Change |\n"
    underage_summary += "|-------|" + "-" * len(label1) + "|" + "-" * len(label2) + "|--------|\n"

    # --- Generate Plots (one per model) ---
    plot_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_keys = selected_models[:2]  # Max 2 models for plotting

    for model_key in model_keys:
        if model_key not in result1["individual_results"] or model_key not in result2["individual_results"]:
            continue

        model_name = multi_classifier.models[model_key]["name"]
        model_short_name = model_name.split('/')[-1]

        # Create a 1x2 figure for this model's comparison
        fig, axes = plt.subplots(1, 2, figsize=(20, 8)) # Wide format for clarity
        fig.suptitle(f'Comparison for {model_short_name}\n{label1} vs {label2}', fontsize=18, fontweight='bold')

        # Get data for both images
        model_result1 = result1["individual_results"][model_key]
        model_result2 = result2["individual_results"][model_key]
        results_for_plot = [model_result1, model_result2]
        underage_results_for_plot = [underage1, underage2]
        labels_for_plot = [label1, label2]
        colors = ['skyblue', 'lightcoral']

        # Find max probability for consistent y-axis
        max_prob = max(
            max(list(model_result1["all_probabilities"].values()) or [0]),
            max(list(model_result2["all_probabilities"].values()) or [0])
        )
        y_max = min(max_prob * 1.2, 1.0)

        # Get underage probabilities and add to summary table
        underage_prob1 = underage1.get("individual_underage", {}).get(model_key, {}).get("underage_probability", 0)
        underage_prob2 = underage2.get("individual_underage", {}).get(model_key, {}).get("underage_probability", 0)
        change = underage_prob2 - underage_prob1
        change_str = f"{change:+.1%}" if change != 0 else "0.0%"
        underage_summary += f"| **{model_short_name}** | {underage_prob1:.1%} | {underage_prob2:.1%} | {change_str} |\n"

        # Plot side-by-side for the current model
        for i, (res, und, lab, col) in enumerate(zip(results_for_plot, underage_results_for_plot, labels_for_plot, colors)):
            ax = axes[i]
            ages = list(res["all_probabilities"].keys())
            probs = list(res["all_probabilities"].values())

            bars = ax.bar(ages, probs, color=col, alpha=0.9, edgecolor='black', linewidth=1.2)
            max_idx = np.argmax(probs)
            bars[max_idx].set_color('darkblue' if i == 0 else 'darkred')
            
            bars[max_idx].set_edgecolor('gold')
            bars[max_idx].set_linewidth(2.5)

            for bar in bars:
                height = bar.get_height()
                if height > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

            predicted_age = res["predicted_age"]
            confidence = res["confidence"]
            underage_prob = und.get("individual_underage", {}).get(model_key, {}).get("underage_probability", 0)

            ax.set_title(f'{lab}\nPredicted: {predicted_age} ({confidence:.1%})', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age Groups', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.6)
            ax.set_ylim(0, y_max)
            ax.text(0.5, -0.35, f'Underage Probability: {underage_prob:.1%}',
                   transform=ax.transAxes, ha='center', va='top', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='yellow', alpha=0.8))

        fig.tight_layout(rect=[0, 0.05, 1, 0.92]) # Adjust layout
        
        # Save the plot for this model
        filename = f"comparison_{model_key}_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(filename)

    # --- Finalize Summary Text ---
    summary += underage_summary

    # Individual model analysis text
    summary += "\n### 📊 Detailed Model Analysis\n"
    for model_key in model_keys:
        if model_key in result1["individual_results"] and model_key in result2["individual_results"]:
            model_name = multi_classifier.models[model_key]["name"]
            model1 = result1["individual_results"][model_key]
            model2 = result2["individual_results"][model_key]
            underage1_prob = underage1.get("individual_underage", {}).get(model_key, {}).get("underage_probability", 0)
            underage2_prob = underage2.get("individual_underage", {}).get(model_key, {}).get("underage_probability", 0)

            summary += f"\n**{model_name}**\n"
            summary += f"- **{label1}:** {model1['predicted_age']} ({model1['confidence']:.1%})\n"
            summary += f"- **{label2}:** {model2['predicted_age']} ({model2['confidence']:.1%})\n"
            summary += f"- **Confidence Change:** {model2['confidence'] - model1['confidence']:+.1%}\n"
            summary += f"- **Underage Probability Change:** {underage2_prob - underage1_prob:+.1%}\n"
            
    # Pad plot_files list to always have 2 elements for the output components
    while len(plot_files) < 2:
        plot_files.append(None)
        
    return plot_files[0], plot_files[1], summary

def create_multi_model_interface():
    """Create multi-model Gradio interface with improved upload handling"""

    with gr.Blocks(title="Multi-Model Age Classification", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Multi-Model Age Classification Tool")
        gr.Markdown("Compare and ensemble multiple age detection models for research and validation")

        with gr.Tab("🆚 Two-Image Comparison"):
            gr.Markdown("### Compare two images with detailed multi-model analysis")
            gr.Markdown("*Perfect for before/after enhancement validation and attention guidance testing*")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**First Image**")
                    comparison_image1 = gr.Image(
                        type="pil",
                        label="Upload First Image",
                        sources=["upload"],
                        interactive=True
                    )
                    comparison_label1 = gr.Textbox(
                        label="Label for First Image",
                        value="",
                        placeholder="Will auto-fill from filename",
                        interactive=True
                    )

                with gr.Column():
                    gr.Markdown("**Second Image**")
                    comparison_image2 = gr.Image(
                        type="pil",
                        label="Upload Second Image", 
                        sources=["upload"],
                        interactive=True
                    )
                    comparison_label2 = gr.Textbox(
                        label="Label for Second Image",
                        value="",
                        placeholder="Will auto-fill from filename",
                        interactive=True
                    )

            with gr.Row():
                comparison_model_selector = gr.CheckboxGroup(
                    choices=["nateraw", "prithiv"],
                    value=["nateraw", "prithiv"],
                    label="Select Models for Comparison",
                    info="Choose models to compare (up to 2 for optimal visualization)"
                )

                comparison_adjustment = gr.Radio(
                    choices=["strict", "balanced"],
                    value="strict",
                    label="Age Boundary Adjustment Mode"
                )

            compare_btn = gr.Button("🔍 Compare Images with Multi-Model Analysis", variant="primary", size="lg")
            
            comparison_plot_1 = gr.Image(label="📈 Comparison Chart 1", type="filepath", container=True, show_label=True)
            comparison_plot_2 = gr.Image(label="📈 Comparison Chart 2", type="filepath", container=True, show_label=True)

            comparison_summary = gr.Markdown(label="📊 Detailed Comparison Analysis")

            # Event handlers for auto-updating labels from filenames
            comparison_image1.upload(
                fn=update_label_from_upload,
                inputs=[comparison_image1],
                outputs=[comparison_label1]
            )
            
            comparison_image2.upload(
                fn=update_label_from_upload,
                inputs=[comparison_image2],
                outputs=[comparison_label2]
            )

            # Main comparison function
            compare_btn.click(
                compare_two_images_multi_model,
                inputs=[comparison_image1, comparison_image2, comparison_label1, comparison_label2, comparison_model_selector, comparison_adjustment],
                outputs=[comparison_plot_1, comparison_plot_2, comparison_summary]
            )
        
        with gr.Tab("📁 Multi-Model Batch Processing"):
            gr.Markdown("### Process entire folders with multi-model analysis and cross-validation")
            
            with gr.Row():
                with gr.Column():
                    batch_folder_input = gr.Textbox(
                        label="📂 Folder Path",
                        placeholder="C:\\path\\to\\your\\images",
                        info="Enter the full path to folder containing images"
                    )
                    
                    batch_model_selector = gr.CheckboxGroup(
                        choices=["nateraw", "prithiv"],
                        value=["nateraw", "prithiv"],
                        label="Select Models for Batch Processing",
                        info="Choose which models to use (more models = better validation)"
                    )
                    
                    with gr.Row():
                        batch_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.1,
                            label="Underage Threshold"
                        )
                        
                        batch_adjustment_mode = gr.Radio(
                            choices=["strict", "balanced"],
                            value="strict",
                            label="🎯 Age Boundary Adjustment Mode",
                            info="Strict: Original model outputs | Balanced: Moderate correction for better accuracy"
                        )
                        
                        batch_output_format = gr.Radio(
                            choices=["csv", "json"],
                            value="csv",
                            label="Output Format",
                            info="CSV for Excel, JSON for detailed data"
                        )
                    
                    batch_btn = gr.Button("🚀 Process Folder with Multi-Model Analysis", variant="primary", size="lg")
                
                with gr.Column():
                    batch_output = gr.Markdown(label="📊 Multi-Model Batch Results")
                    batch_download_file = gr.File(label="📥 Download Full Results", visible=True)
                    batch_status = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown("### 📊 Batch Processing Summary Table")
            gr.Markdown("Sorted by Ensemble Underage Probability. Probabilities are color-coded based on the threshold.")
            
            summary_table = gr.Dataframe(
                label="Batch Summary",
                headers=[
                    "filename",
                    "ensemble_underage_probability",
                    "nateraw_underage_probability",
                    "nateraw_distribution_type",
                    "prithiv_underage_probability",
                    "prithiv_distribution_type"
                ],
                interactive=False,
            )
            
            batch_btn.click(
                batch_process_folder_multi,
                inputs=[batch_folder_input, batch_model_selector, batch_threshold, batch_adjustment_mode, batch_output_format],
                outputs=[batch_output, batch_download_file, batch_status, summary_table]
            )

        with gr.Tab("ℹ️ Multi-Model Guide & Research Workflow"):
            gr.Markdown("""
            ## 🤖 Multi-Model Age Classification Guide
            
            ### 📋 Key Features
            
            - **Dual Model Support**: nateraw/vit-age-classifier and prithivMLmods/Age-Classification-SigLIP2
            - **Ensemble Analysis**: Combines predictions from multiple models for better accuracy
            - **Auto-Filename Labels**: Uploaded filenames automatically populate comparison labels
            - **Advanced Statistics**: Distribution analysis and model agreement metrics
            
            ### 🔄 Comparison Workflow
            
            1. **Upload Images**: Files are uploaded with improved stability
            2. **Auto-Labels**: Filenames automatically populate label fields (editable)
            3. **Model Selection**: Choose which models to use for analysis
            4. **Adjustment Mode**: Select age boundary correction level
            5. **Generate Analysis**: Get detailed multi-model comparison charts
            
            ### 🧠 Model Architectures
            
            - **nateraw**: Vision Transformer (ViT) with 9 age groups
            - **prithiv**: SigLIP with 5 age categories  
            
            ### 🎯 Age Boundary Adjustments
            
            - **Strict**: Uses original model outputs without any adjustments
            - **Balanced**: Applies moderate corrections for better age boundary accuracy, particularly for teenage classification
            
            ### 🔬 Research Applications
            
            - **Architecture Comparison**: CNN vs Transformer approaches
            - **Training Scale Effect**: Small datasets vs massive pre-training
            - **Ensemble Methods**: How different model types complement each other
            - **Age Boundary Analysis**: Understanding model biases and corrections
            
            ### 📊 Understanding the Results
            
            - **Individual Results**: Each model's predictions and confidence scores
            - **Harmonized Groups**: Standardized age categories for fair comparison
            - **Ensemble Prediction**: Combined prediction from multiple models
            - **Agreement Analysis**: How well models agree on predictions
            """)

    return demo

def main():
    """Main function for multi-model classifier"""
    print("Starting Multi-Model Age Classification App...")

    # Load default model
    if not multi_classifier.load_model("nateraw"):
        print("Failed to load default model. Exiting.")
        sys.exit(1)

    # Try to load additional models (optional)
    print("Attempting to load additional models...")
    multi_classifier.load_model("prithiv")

    # Create and launch interface
    demo = create_multi_model_interface()

    print("Launching multi-model web interface...")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()