# sign-language-mnist-ml

# Sign Language MNIST - Machine Learning Assignments

## Overview
Two machine learning assignments on the 
[Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) 
for American Sign Language (ASL) hand gesture recognition.

**Dataset:** 27,455 grayscale 28×28 images, 24 classes (letters A-Y, excluding J and Z)

---

## Assignment 1 - Classification
**Goal:** Develop and evaluate classification algorithms for ASL hand sign recognition.

### Models Implemented
- k-Nearest Neighbors (kNN)
- Support Vector Machines (SVM)
- Decision Trees
- Random Forest ← Best Model (88% on test set)
- AdaBoost
- Feedforward Neural Networks (FNN)
- Convolutional Neural Networks (CNN) via PyTorch

### Results
| Model | Test Accuracy |
|-------|--------------|
| kNN | 84% |
| SVM (Linear) | 84% |
| Random Forest | **88%** |
| FNN | 76% |

### Hidden Phrase Challenge
A `predict_letters()` function was implemented using the best model (Random Forest) 
wrapped in a scikit-learn Pipeline for the class competition.

---

## Assignment 2 - Representation Learning, Clustering & Generative Models
**Goal:** Feature extraction with deep neural networks, clustering, and generative modeling.

### Part A - CNN Autoencoder + Clustering
- CNN Autoencoder with latent dimensions M = {2, 10, 50}
- Classification in latent space using best model from Assignment 1
- K-means clustering with Silhouette Score
- Hierarchical Agglomerative Clustering with dendrogram

### Part B - Generative Models
- Variational Autoencoder (VAE)
- Conditional VAE (cVAE)
- Comparison: VAE vs cVAE vs PCA

### Key Finding
> *"The VAE latent space is more suitable for clustering than the AE latent space 
> because the KL Divergence constraint forces it to follow N(0,I), 
> creating a continuous and organized space without gaps between clusters."*

---

## Requirements

scikit-learn==1.8.0
torch
pandas
numpy
matplotlib
scipy

---

## Course
Machine Learning  
Department of Computer Science and Engineering
