# Multi-Model Age Classification Research Tool

A comprehensive research framework for comparing and ensembling multiple deep learning models for age estimation from facial images. This tool enables detailed analysis of different architectural approaches and provides insights into model agreement, confidence distributions, and age boundary detection.

![Multi-Model Comparison](https://img.shields.io/badge/Models-2%20Architectures-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Research](https://img.shields.io/badge/Purpose-Research-purple)

## 🎯 Overview

This project implements a multi-model approach to age classification, comparing Vision Transformer (ViT) and SigLIP architectures to provide robust age estimation with statistical analysis. Originally developed for research into model agreement and ensemble methods for age detection applications.

### Key Features

- **🤖 Dual Architecture Comparison**: Vision Transformer vs SigLIP models
- **📊 Statistical Analysis**: Distribution analysis, entropy calculations, and confidence metrics
- **🔄 Ensemble Methods**: Model agreement analysis and combined predictions
- **⚙️ Adjustable Age Boundaries**: Strict vs balanced age threshold detection
- **📁 Batch Processing**: Folder-level analysis with progress tracking
- **📈 Visual Comparisons**: Side-by-side model performance charts
- **💾 Export Capabilities**: CSV and JSON output formats

## 🧠 Model Architectures

| Model | Architecture | Age Groups | Training Data | Strengths |
|-------|-------------|------------|---------------|-----------|
| **nateraw/vit-age-classifier** | Vision Transformer (ViT) | 9 categories (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+) | ImageNet pre-trained | Fine-grained age discrimination |
| **prithivMLmods/Age-Classification-SigLIP2** | SigLIP | 5 categories (Child 0-12, Teenager 13-20, Adult 21-44, Middle Age 45-64, Aged 65+) | Large-scale multimodal training | Robust feature extraction |

## 🚀 Quick Start

### One-Click Installation (Recommended)

1. **Download the installer**
   - Right-click → "Save As": [install_age_classifier_batch.bat](https://github.com/kvmierlo3/age_classifier/raw/main/install_age_classifier_batch.bat) **(Version 2.0 FINAL)**
   - Place it in your desired installation directory

2. **Run the installer**
```bash
# Double-click or run in command prompt
install_age_classifier_batch.bat
```

The installer will automatically:
- Clone the repository
- Set up Python virtual environment
- Install all dependencies
- Download pre-trained models
- Create desktop shortcuts

3. **Start the application**
```bash
# Double-click or run
start_age_classifier_batch.bat
```

4. **Access the web interface**
```
http://127.0.0.1:7861
```

### Manual Installation (Advanced Users)

```bash
git clone https://github.com/kvmierlo3/age_classifier.git
cd age_classifier
pip install -r requirements.txt
python multi_model_age_classifier_cleaned.py
```

## 📖 Usage

### Two-Image Comparison

Perfect for research validation and before/after analysis:

1. Upload two images for comparison
2. Select models to analyze (nateraw, prithiv, or both)
3. Choose age boundary adjustment mode:
   - **Strict**: Original model outputs without adjustments
   - **Balanced**: Moderate corrections for better teenage classification accuracy
4. Generate detailed comparison charts and statistics

### Batch Processing

For large-scale research analysis:

1. Specify folder path containing images
2. Configure processing parameters
3. Export results as CSV or JSON
4. Analyze summary statistics and model agreement patterns

## 🔬 Research Applications

### Architecture Comparison Studies
- **Transformer vs CNN-based approaches**: Compare ViT and SigLIP performance
- **Training paradigm analysis**: Different pre-training strategies and their impact
- **Age group granularity**: 9-category vs 5-category classification effectiveness

### Ensemble Method Research
- **Model agreement analysis**: Quantify consensus across different architectures
- **Confidence calibration**: Study prediction certainty across models
- **Harmonized predictions**: Unified age group mapping for fair comparison

### Age Boundary Analysis
- **Threshold sensitivity**: Impact of different age boundary adjustments
- **Teenage classification**: Addressing the challenging 13-20 age range
- **Underage detection**: Research applications requiring age verification

## 📊 Statistical Metrics

The tool provides comprehensive statistical analysis:

- **Entropy**: Measure of prediction uncertainty
- **Gini Coefficient**: Concentration of probability mass
- **Peak Ratio**: Dominance of highest probability
- **Distribution Types**: Confident, Moderate, Uncertain, Very Uncertain
- **Agreement Quality**: Strong Agreement, Harmonized Agreement, Disagreement

## 🛠️ Technical Details

### Dependencies
```
torch>=2.0.0
transformers>=4.30.0
gradio>=4.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
```

### Architecture
- **Backend**: PyTorch-based model inference
- **Frontend**: Gradio web interface
- **Processing**: Multi-threaded batch processing with progress tracking
- **Export**: Pandas-based CSV generation with styled output

### Model Integration
Both models are loaded dynamically and can be used independently or in ensemble:

```python
# Load individual models
classifier.set_active_models(["nateraw"])  # ViT only
classifier.set_active_models(["prithiv"])  # SigLIP only
classifier.set_active_models(["nateraw", "prithiv"])  # Both models

# Classify with ensemble analysis
result = classifier.classify_age_multi_model(image, include_ensemble=True)
```

## 📈 Example Results

### Individual Model Output
```python
{
    "model": "nateraw",
    "predicted_age": "20-29",
    "confidence": 0.847,
    "distribution_analysis": {
        "distribution_type": "Confident",
        "entropy": 0.234,
        "peak_ratio": 4.2
    }
}
```

### Ensemble Analysis
```python
{
    "ensemble_result": {
        "predicted_age": "Young Adult (20-29)",
        "confidence": 0.823,
        "method": "average"
    },
    "agreement_analysis": {
        "agreement_quality": "Strong Agreement",
        "confidence_std": 0.023
    }
}
```

## 🎯 Age Boundary Adjustment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Strict** | Original model outputs without modification | Baseline comparison, model evaluation |
| **Balanced** | Moderate teenage probability adjustment (0.9x nateraw, 0.8x prithiv) | Practical applications, improved accuracy |

## 📁 File Structure

```
age_classifier/
├── multi_model_age_classifier_cleaned.py  # Main application
├── requirements.txt                       # Dependencies
├── install_age_classifier_batch.bat      # One-click installer (v2.0 FINAL)
├── start_age_classifier_batch.bat        # Application launcher (v2.0 FINAL)
├── age_classifier.md                     # This documentation
└── examples/                             # Example outputs
    ├── comparison_charts/                # Sample comparison visualizations
    └── batch_results/                    # Example batch processing outputs
```

## 🤝 Contributing

Contributions are welcome! This project is designed for research applications. Please consider:

1. **Model Integration**: Adding new age classification models
2. **Statistical Metrics**: Implementing additional analysis methods
3. **Visualization**: Enhancing chart generation and comparison tools
4. **Performance**: Optimizing batch processing for larger datasets

### Development Setup
```bash
git clone https://github.com/kvmierlo3/age_classifier.git
cd age_classifier
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **nateraw** for the ViT-based age classifier
- **prithivMLmods** for the SigLIP-based age classification model
- **Hugging Face** for the transformers library and model hosting
- **Gradio** for the intuitive web interface framework

## 📚 Citations

If you use this tool in your research, please cite:

```bibtex
@software{age_classifier,
  title={Multi-Model Age Classification Research Tool},
  author={kvmierlo3},
  year={2024},
  url={https://github.com/kvmierlo3/age_classifier}
}
```

## ⚠️ Research Ethics

This tool is intended for research purposes. When using for age-related applications:

- Ensure compliance with privacy regulations
- Consider bias and fairness in age classification systems
- Validate results with appropriate ground truth data
- Follow ethical guidelines for computer vision research

## 🔗 Related Work

- [nateraw/vit-age-classifier](https://huggingface.co/nateraw/vit-age-classifier)
- [prithivMLmods/Age-Classification-SigLIP2](https://huggingface.co/prithivMLmods/Age-Classification-SigLIP2)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [SigLIP Paper](https://arxiv.org/abs/2303.15343)

---

**Note**: This project focuses on comparing established, well-documented models for research reproducibility and transparency. The tool provides honest, evidence-based analysis without exaggerated claims about model capabilities.
