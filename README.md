# 🧠 Mini-Transformer from Scratch (PyTorch)

A deep-dive AI/ML project that abandons high-level libraries (like HuggingFace) to build the mathematical core of Large Language Models from scratch using raw **PyTorch** tensors.

## 🎯 AIML Objectives
This project focuses 100% on Deep Learning model architecture and tensor mathematics:
- **Multi-Head Self-Attention:** Implementing the Scaled Dot-Product Attention mechanism ($Softmax(\\frac{QK^T}{\\sqrt{d_k}})V$).
- **The Transformer Block:** Building the core architecture including Layer Normalization, Feed-Forward Networks, and Residual (Skip) Connections.
- **Einsum Optimization:** Using Einstein Summation (`torch.einsum`) for highly optimized, multi-dimensional matrix multiplication.

## 🛠 Tech Stack
- **Python 3.12**
- **PyTorch** (`torch`, `torch.nn`)
- **NumPy & Math**

## 🚀 How to Run
1. Open the Jupyter Notebook `transformer_from_scratch.ipynb`.
2. Run the cells sequentially to build the `SelfAttention` and `TransformerBlock` PyTorch modules.
3. The final cell generates a fake tensor (simulating embedded text tokens) and passes it through the AI architecture to verify tensor shape transformations.

## 💡 What I Learned (Interview Highlights)
- **Self-Attention:** Discovered exactly how an LLM mathematically compares every word in a sequence to every other word simultaneously to generate contextual Queries, Keys, and Values.
- **Skip Connections:** Implemented residual connections to prevent the vanishing gradient problem in deep neural networks.
- **PyTorch Tensor Manipulation:** Mastered reshaping and transposing 4-dimensional tensors (Batch Size, Sequence Length, Attention Heads, Head Dimension) for parallelized GPU training.
