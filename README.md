# 🧩 Enigma Match: Context-Aware Compatibility Prediction System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![Status](https://img.shields.io/badge/Status-Hackathon%20Complete-green)

## 📖 Overview
**Enigma Match** is a machine learning solution designed to solve the "Cold Start" problem in professional networking. Unlike simple recommendation systems that rely on cosine similarity, this project implements a **Context-Aware Compatibility Regression Network (CACRN)**. 

It predicts a compatibility score (0 to 1) between two users by analyzing their **Business Objectives**, **Interests**, and **Constraints** using a hybrid "Wide & Deep" architecture.

## 🧠 Model Architecture
The core of the system is a **Siamese Neural Network** with a custom Interaction Head.



### Key Components:
1.  **Deep Path (The "Vibe" Check):**
    * Uses **Sentence-BERT (all-MiniLM-L6-v2)** to generate dense vector embeddings for unstructured text (Interests, Objectives).
    * Processes User A and User B through shared weight encoders (Siamese Towers).
2.  **Interaction Layer:**
    * Computes explicit relationship vectors:
        * **Dot Product:** Measures alignment direction.
        * **Absolute Difference:** Measures demographic distance (Age, Seniority).
        * **Element-wise Multiplication:** Captures non-linear feature interactions.
3.  **Wide Path (The "Logic" Check):**
    * Injects handcrafted engineering features directly into the final classification layer.
    * **Features:** Hard constraint violations (e.g., "No Finance" vs. "Finance Industry"), interest overlaps, and role complementarity.



## 🛠️ Technical Highlights
* **Semantic Understanding:** Replaced naive keyword matching with Transformer-based embeddings (SBERT) to understand context (e.g., "ML Engineer" $\approx$ "Data Scientist").
* **Imbalance Mitigation:** Implemented **Aggressive Sample Weighting** (20x-50x penalty) to force the model to learn rare high-compatibility matches in a dataset dominated by zeros.
* **Robust Loss Function:** Utilized **Mean Absolute Error (MAE)** to sharpen predictions and prevent regression to the mean.

## 🚀 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/Enigma-Match-ML.git](https://github.com/yourusername/Enigma-Match-ML.git)
    cd Enigma-Match-ML
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    *Note: For GPU support on Windows Native, ensure you use TensorFlow < 2.11.*
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

### 1. Training the Model
To train the model from scratch using the provided dataset:
```bash
python main.py --mode train --train-profiles data/train.xlsx --target data/target.csv --epochs 50 --batch-size 128
