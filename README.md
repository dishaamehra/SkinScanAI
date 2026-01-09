## SkinScanAI — Explainable Deep Learning for Skin Lesion Classification

    SkinScanAI is an AI-powered skin lesion classification system built using deep learning and ensemble learning, designed to assist early detection of skin cancer.
    The project focuses not only on accuracy, but also on model interpretability using Grad-CAM, making predictions more transparent and clinically meaningful.

## Problem Motivation
    Skin cancer is one of the most common cancers worldwide, yet early diagnosis remains challenging due to:
  
    Visual similarity between different lesion types
  
    Subjective interpretation by clinicians
  
    Limited access to dermatologists in many regions
  
    SkinScanAI explores how deep learning models can support dermatologists by providing consistent, explainable, and high-performing predictions on dermoscopic images.
## Key Features

    Multi-class skin lesion classification (7 classes)
    
    Comparative study of state-of-the-art CNNs
    
    1.ResNet50
    
    2.EfficientNetB0
    
    3.InceptionV3
    
    4.Ensemble learning (soft-voting) for improved reliability
    
    Grad-CAM visual explanations highlighting lesion-relevant regions
    
    Class imbalance handling using weighted loss functions
    
    Comprehensive evaluation using accuracy, precision, recall, F1-score, and confidence analysis
## Model Architecture

    Each model is fine-tuned using ImageNet pre-trained weights and trained on dermoscopic images.
    
    Individual Models
    
    ResNet50 — strong feature extraction with residual connections
    
    EfficientNetB0 — lightweight and efficient, suitable for real-world deployment
    
    InceptionV3 — captures multi-scale lesion patterns
    
    Ensemble Model
    
    Predictions from all three models are averaged (soft voting) to reduce variance and improve generalization.
## Dataset

    HAM10000 (Kaggle)
    
    10,015 dermoscopic images
    
    7 skin lesion classes:
    
      1.Melanocytic Nevus
      
      2.Melanoma
      
      3.Benign Keratosis
      
      4.Basal Cell Carcinoma
      
      5.Actinic Keratoses
      
      6.Vascular Lesions
      
      7.Dermatofibroma
    
    >> Dataset is highly imbalanced — addressed using weighted cross-entropy loss and data augmentation.

## Results
<img width="703" height="437" alt="image" src="https://github.com/user-attachments/assets/6fcd1d8d-1fcb-4374-bd08-ec5be0d1725e" />

    Ensemble model achieved the best balance across all classes
    
    Improved performance on minority lesion categories
    
    More stable confidence distribution compared to individual models
## Explainability with Grad-CAM

    To address the black-box nature of deep learning:
    
    Grad-CAM heatmaps visualize which regions influenced predictions
    
    Models consistently focus on lesion regions instead of background
    
    Improves trust, transparency, and clinical interpretability

## Tech Stack

    Python
    
    PyTorch
    
    CNNs (ResNet, EfficientNet, Inception)
    
    Grad-CAM
    
    NumPy, Matplotlib, OpenCV
    
    CUDA (GPU training)
    
## Disclaimer

    This project is intended for research and educational purposes only.
    It is not a medical diagnostic tool and should not replace professional medical advice.

## Future Work

    Cross-dataset validation for better generalization
    
    Integration with clinical metadata (age, lesion location, history)
    
    Uncertainty estimation in predictions
    
    Deployment as a web or mobile screening tool
