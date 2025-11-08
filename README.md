# Neural Network from Scratch using Custom Autograd Engine

![MLP Training](https://miro.medium.com/v2/1*gMJz6v4nQNXXxbDgYuynGg.gif )

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-brightgreen.svg)](https://github.com/RohankumarReddy/ScratchMLP56)
---

## Project Overview

This project demonstrates the implementation of a **fully-connected neural network (MLP)** from scratch using a custom `Value` class for automatic differentiation. The notebook showcases the core principles of forward and backward propagation without relying on external deep learning libraries like PyTorch for gradient computation.

The implementation provides an educational insight into how neural networks learn by manually computing gradients and updating parameters, offering a foundational understanding of **autograd, backpropagation, and gradient descent**.

---

## Features

- Custom **`Value` class** supporting arithmetic operations (`+`, `-`, `*`, `/`, `**`), `tanh`, and `exp` with gradient tracking.
- Support for **multiple layers and neurons** in an MLP.
- Manual **forward pass** computation for predictions.
- Implementation of **mean squared error (MSE)** loss function.
- Backpropagation using **custom gradient computation**.
- Parameter updates via **gradient descent**.

---

## Notebook Contents

1. **Value Class**
   - Handles scalar values and builds a computation graph.
   - Supports standard operations with automatic gradient computation.
   - Implements a `backward()` method to perform backpropagation.

2. **Neuron, Layer, and MLP Classes**
   - `Neuron`: Single neuron with input weights and bias.
   - `Layer`: Collection of neurons forming a single layer.
   - `MLP`: Multi-layer perceptron with configurable hidden layers.

3. **Dataset**
   - Sample dataset with input features (`xs`) and target outputs (`ys`).

4. **Training Loop**
   - Forward pass to compute predictions.
   - Loss computation using MSE.
   - Backward pass to compute gradients.
   - Gradient descent updates to train the network.

5. **Results**
   - Convergence of loss over training iterations.
   - Predictions of the trained MLP on sample inputs.

---

## Installation & Requirements

- Python 3.8+
- Standard Python libraries: `math`, `random`
- Optional: **Jupyter Notebook** or **Google Colab** for execution.

---

## Usage

1. Clone the repository or download the notebook.
2. Open the notebook in **Jupyter Notebook** or **Google Colab**.
3. Execute the cells sequentially:
   - Define the `Value` class.
   - Initialize neurons, layers, and the MLP.
   - Prepare dataset (`xs` and `ys`).
   - Run the training loop to minimize loss.
4. Observe predictions and loss convergence.

---

## Key Learnings

- Manual understanding of **forward and backward propagation**.
- Hands-on implementation of **automatic differentiation**.
- Insight into how **neural networks update weights** using gradients.
- Reinforcement of **MLP architecture concepts**, including neurons, layers, activations, and loss computation.

---


## Notes

- This implementation is primarily educational and **not optimized for large-scale data**.
- All computations are scalar-based for clarity; for industrial-scale applications, libraries like PyTorch or TensorFlow are recommended.
- The notebook demonstrates **core machine learning concepts** suitable for interviews, coursework, or self-study.
