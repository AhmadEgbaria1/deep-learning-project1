# deep-learning-project1
 MNIST & XOR

This repository contains the implementation and analysis for Assignment #1 of the Deep Learning course. The project explores the transition from classical Machine Learning to Deep Networks, focusing on multiclass logistic regression, multi-layer perceptrons (MLP), weight initialization, and manual backpropagation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Part 1: MNIST Classification](#part-1-mnist-classification)
   - [Logistic Regression vs. MLP](#logistic-regression-vs-mlp)
   - [Weight Initialization Analysis](#weight-initialization-analysis)
   - [Optimization Methods](#optimization-methods)
3. [Part 2: Multi-dimensional XOR](#part-2-multi-dimensional-xor)
   - [Implementation from Scratch](#implementation-from-scratch)
   - [Universal Approximation Theorem](#universal-approximation-theorem)
   - [Manual Backpropagation](#manual-backpropagation)

## Project Overview
The assignment is divided into two main sections:
- **Question 1:** Comparing a multiclass logistic regression baseline with a Feedforward Neural Network (FFNN) on the MNIST dataset. It includes an analysis of how initialization and optimization affect model convergence.
- **Question 2:** Modeling the XOR problem using a custom-built FFNN (built without `nn.Linear`) and verifying manual gradient computations against PyTorch's `autograd`.

## Part 1: MNIST Classification

### Logistic Regression vs. MLP
- **Baseline:** A multiclass logistic regression reached ~92% accuracy. 
- **MLP Architecture:** A two-layer MLP (784-400-400-10) with ReLU activation reached ~98% accuracy.
- **Finding:** The non-linear activation functions and hidden layers allow the MLP to learn complex spatial relationships that a linear hyperplane cannot separate.

### Weight Initialization Analysis
We compared four initialization schemes:
- **Zero:** Fails to train due to symmetry (all neurons receive the same gradient).
- **Uniform [0, 1]:** Highly unstable, poor performance.
- **Normal (Standard):** Functional but slower convergence.
- **Xavier:** Best performance for stabilizing activations and gradients.

### Optimization Methods
Compared the performance of **Adam**, **SGD**, **RMSProp**, and **Adagrad**. 
- **Key Insight:** Initialization is often more critical than the optimizer choice; even a 'best' optimizer like Adam cannot rescue a model initialized with zeros.

## Part 2: Multi-dimensional XOR

### Implementation from Scratch
Implemented a `Linear` layer and `FFNet` using only raw PyTorch tensors and `nn.Parameter` to understand the underlying mechanics of neural network layers.

### Universal Approximation Theorem
Tested dimensions $d=2$ through $d=5$ to see how hidden layer size affects convergence. 
- **Result:** While the theorem guarantees a solution exists, convergence is highly sensitive to initialization. Models with $d=4$ and $d=5$ required significantly more capacity or better luck with random seeds to converge within 300 epochs.

### Manual Backpropagation
Developed `calc_gradients` to manually compute the partial derivatives of the MSE loss with respect to all weights and biases using the chain rule. Correctness was verified by comparing results to `loss.backward()`.

## Setup & Requirements
- Python 3.x
- PyTorch
- Matplotlib, NumPy, Torchvision

```python
# Clone the repo
git clone https://github.com/your-username/deep-learning-assignment-1.git
