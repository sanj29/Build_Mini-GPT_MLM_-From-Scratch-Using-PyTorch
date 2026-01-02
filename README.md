# 🧠 NanoGPT – Build a Mini GPT from Scratch (PyTorch)

A **minimal, educational implementation of a GPT-style language model** built from scratch using PyTorch.

This repository is designed to help engineers and beginners **understand how Large Language Models (LLMs actually work internally)** — without abstractions, frameworks, or hidden magic.

---

## 🚀 What You Will Learn

By going through this repository, you will understand:

- How **GPT-style autoregressive models** work
- How text is converted into **tokens and embeddings**
- How **self-attention** enables context-aware predictions
- How models learn using **next-token prediction**
- How text is **generated one token at a time**
- Why small datasets lead to **overfitting and early termination**

This is **core GPT logic**, simplified for learning.

---

## 📁 Project Structure

```
nano-gpt/
│
├── main.py                 # Training + generation script
├── transformer_blocks.py   # Transformer block (Attention + FFN)
├── README.md               # Documentation
```

---

## 📚 Dataset

The model is trained on a **small word-level dataset** for educational purposes.

Each sentence is terminated with a special token:

```
<END>
```

Example:

```
Delhi has foggy weather, see you in Delhi <END>
Love to see you all soon, let's meet in Bangalore <END>
```

The `<END>` token teaches the model **when to stop generating text**.

⚠️ Note: Small datasets intentionally cause **memorization**, which is useful for learning.

---

## 🔤 Tokenization (Word-Level)

A simple word-level tokenizer is used:

```python
words = sorted(set(text.split()))
words2index = {word: index}
idx2words = {index: word}
```

---

## 🧮 Training Data Format

The model learns using **next-token prediction**.

For a context length (`block_size`) of 6:

```
Input  (x): [Delhi, has, foggy, weather, see, you]
Target (y): [has, foggy, weather, see, you, in]
```

---

## 🧠 Model Architecture (NanoGPT)

### 1️⃣ Token Embeddings
Each word is mapped to a dense vector using an embedding layer.

### 2️⃣ Positional Embeddings
Provide sequence order information to the model.

### 3️⃣ Transformer Blocks
Contain self-attention, feed-forward networks, residual connections, and normalization.

### 4️⃣ Language Modeling Head
Maps hidden states to vocabulary probabilities.

---

## 📉 Loss Function

The model is trained using **CrossEntropyLoss** to predict the next token.

---

## ✍️ Text Generation (Autoregression)

Text is generated **one token at a time**, feeding the output back as input.

---

## 🧪 Example Output

```
Input Prompt:
Delhi

Generated Text:
Delhi has foggy weather see you in Delhi
```

---

## ⚠️ Limitations (By Design)

- Small dataset
- Word-level tokenization
- No temperature sampling
- No dropout

---

## 🏁 Final Notes

This project focuses on **clarity over scale**.

If you understand this repository, you understand how GPT works internally.


