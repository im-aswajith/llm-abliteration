<div align="center">

# 🧠 LLM Abliteration

### Advanced Hidden-State Abliteration Framework for Transformer Language Models

<p align="center">
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=24&pause=1000&color=00F5FF&center=true&vCenter=true&width=900&lines=Advanced+LLM+Abliteration+Framework;Hidden+State+Vector+Manipulation;Transformer+Research+Toolkit;Fast+%7C+Modular+%7C+Python+Powered" />
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)

![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)

![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch)

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

![Research](https://img.shields.io/badge/Research-LLM-purple?style=for-the-badge)

</p>

---

### ⚡ Modify Hidden Representations Instead of Retraining

*A lightweight research framework for experimenting with hidden-state direction manipulation in Transformer-based language models.*

</div>

---

# 📖 Overview

**LLM Abliteration** is an experimental Python framework for modifying transformer language models by analyzing and manipulating hidden-state representations.

Unlike traditional fine-tuning, this project performs vector-based modifications inside transformer hidden representations to alter specific behavioral directions while preserving the original model architecture.

The framework focuses on:

- Hidden-state extraction
- Direction analysis
- Layer-wise manipulation
- Projection techniques
- Automatic optimization
- Model export

Because weights are modified directly after analysis, no gradient-based training loop is required.

---

# ✨ Features

## 🧠 Hidden-State Analysis

- Extract intermediate transformer representations
- Analyze hidden activations layer-by-layer
- Multi-layer processing
- Configurable layer selection

---

## ⚡ Advanced Direction Manipulation

- Norm preserving projection
- Bi-projection support
- Multi-direction processing
- Triangular layer falloff
- Layer-aware modifications

---

## 🎯 Automatic Optimization

- Alpha optimization
- Configurable strength
- Direction scaling
- Layer-specific optimization

---

## ⚙️ Flexible Configuration

- Custom prompts
- Adjustable batch size
- Maximum sequence length
- Optional 4-bit loading
- BF16 support
- Automatic device mapping

---

## 🚀 Efficient Processing

- No traditional fine-tuning
- No optimizer
- No gradient descent
- Direct weight manipulation
- Hugging Face compatible

---

# 🛠️ Requirements

- Python 3.10+
- PyTorch
- Transformers
- NumPy
- SciPy
- tqdm

---

# 📦 Installation

Clone the repository

```bash
git clone https://github.com/USERNAME/llm-abliteration.git

cd llm-abliteration
```

Install dependencies

```bash
pip install torch transformers scipy numpy tqdm
```

---

# 🚀 Usage

Run the main abliteration script

```bash
python advanced_abliterate.py
```

If you have ≥12 GB VRAM (full precision, faster)
```bash
python advanced_abliterate.py --num-samples 64 --verbose
```

If you have 8–12 GB VRAM (use 4‑bit to save memory)

```bash
python advanced_abliterate.py --load-4bit --num-samples 64 --verbose
```

If you have ≤8 GB VRAM

```bash
python advanced_abliterate.py --load-4bit --batch-size 2 --num-samples 32 --verbose
```



Chat with the exported model

```bash
python chat_with_abliterated.py
```

---

# ⚙️ Configuration

Most settings are located inside:

```python
AdvancedAbliterationConfig
```

Important options include:

```python
model_id

output_dir

harmful_prompts

harmless_prompts

batch_size

num_samples

max_length

alpha

layers

load_in_4bit
```

You can replace the prompt datasets with your own research datasets depending on the experiment.

---

# 🔬 Workflow

```text
                Load Model
                     │
                     ▼
          Extract Hidden States
                     │
                     ▼
        Analyze Representation Space
                     │
                     ▼
       Compute Direction Vectors
                     │
                     ▼
     Optimize Projection Strength
                     │
                     ▼
     Apply Layer-wise Modification
                     │
                     ▼
         Save Modified Model
                     │
                     ▼
      Chat / Evaluate / Experiment
```

---

# 📂 Project Structure

```
llm-abliteration/

│
├── advanced_abliterate.py
│
├── chat_with_abliterated.py
│
├── LICENSE
│
└── README.md
```

---

# 🧠 Core Concepts

This framework operates using hidden-state representation analysis.

General workflow:

1. Load a pretrained Transformer model.
2. Collect hidden activations.
3. Compute representation directions.
4. Optimize projection strength.
5. Modify selected transformer layers.
6. Save the resulting model.

Unlike supervised fine-tuning, the process does not involve iterative gradient updates over multiple epochs.

---

# ⚡ Advantages

- Lightweight experimentation
- Fast execution
- Minimal dependencies
- Modular architecture
- Easy customization
- Hugging Face compatible
- Layer-wise control
- Python only

---

# 🔧 Customization

You can customize:

- Prompt datasets
- Model selection
- Projection strength
- Target layers
- Batch size
- Sequence length
- Output directory

without modifying the core architecture.

---

# 📈 Research Applications

This repository can be used for exploring topics such as:

- Hidden-state representation analysis
- Transformer interpretability
- Layer-wise behavior modification
- Directional activation analysis
- Representation engineering
- Experimental LLM research

---

# ⚠️ Notes

- This project is intended for research and educational purposes.
- Performance depends on the chosen model and configuration.
- Larger models require significantly more memory.
- GPU acceleration is recommended for faster processing.

---

# 📜 License

This project is licensed under the **MIT License**.

See the **LICENSE** file for full details.

---

# ❤️ Contributing

Contributions are welcome.

If you'd like to improve the project:

- Fork the repository
- Create a new branch
- Commit your changes
- Submit a Pull Request

---

# ⭐ Support

If this project helped you, consider giving it a **⭐ Star**.

It helps the project reach more developers and supports future improvements.

---

<div align="center">

## ⭐ Thanks for Visiting!

### Happy Researching 🚀

Made with ❤️ using Python & Transformers

</div>
