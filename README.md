# cinegraph-recommender
# 🎬 CineGraph: LightGCN-based Movie Recommender System

An end-to-end movie recommendation system built using **LightGCN**, enhanced with **semantic embeddings (SBERT)** and an **LLM-powered conversational interface**.

> Based on: *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (SIGIR 2020)

---

## 🚀 Overview

Recommender systems often struggle with sparse user interactions.
This project models recommendations as a **graph problem**, where users and items are connected through interactions.

We:

* Implement LightGCN **from scratch**
* Extend it with **semantic understanding (SBERT)**
* Integrate an **LLM-based chatbot interface**
* Explore **hybrid recommendation strategies**

---

## 🧠 Key Contributions

* ✅ Reproduced LightGCN with improved performance (**NDCG@10 = 0.4015 > 0.3891**) 
* ✅ Compared against MF, NCF, NGCF under a unified evaluation protocol
* ✅ Proposed **LightGCN + SBERT** for semantic initialization
* ✅ Evaluated **cold-start scenarios**
* ✅ Built a **Streamlit chatbot interface using LLM**
* ✅ Explored **hybrid recommendation (LightGCN + LLM)**

---

## 📊 Results

### 🔹 Collaborative Filtering Models (Full Dataset)

| Model                | HR@10     | NDCG@10   |
| -------------------- | --------- | --------- |
| MF                   | 0.669     | 0.406     |
| NCF                  | 0.140     | 0.069     |
| NGCF                 | 0.666     | 0.404     |
| **LightGCN (ours)**  | **0.682** | **0.401** |
| **LightGCN + SBERT** | **0.685** | **0.404** |

👉 LightGCN outperforms NGCF despite being simpler
👉 SBERT provides **consistent improvements**

---

### 🔹 LLM & Hybrid Models (200-user sample)

| Model                       | HR@10    | NDCG@10  |
| --------------------------- | -------- | -------- |
| LLM Recommender             | 0.71     | 0.31     |
| **Hybrid (LightGCN + LLM)** | **0.80** | **0.53** |

👉 Hybrid achieves highest performance (experimental setting)

---

## ❄️ Cold-Start Insight

SBERT improves recommendations when user interaction data is limited.

👉 Key takeaway:

> Semantic information helps overcome sparsity.

---

## ⚙️ Model Overview

### LightGCN

* Graph-based collaborative filtering
* Removes transformations & nonlinearities
* Uses **pure neighborhood aggregation**

**Propagation:**

$$
E^{(k)} = \hat{A} E^{(k-1)}
$$

**Final embedding:**

$$
E = \frac{1}{K+1} \sum_{k=0}^{K} E^{(k)}
$$

---

## 🤖 LLM Integration

* Enables **natural language interaction**
* Provides **explanations for recommendations**
* Used in **hybrid recommendation experiments**

👉 Important:

> LLM enhances interaction — LightGCN performs ranking

---

## 🖥️ Demo

The system supports conversational queries like:

> “I like action movies, not too gory”

And returns:

* 🎬 Movie recommendations
* 💬 Natural language explanations

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 📁 Project Structure

```
cinegraph-recommender/
├── app/            # Streamlit chatbot interface
├── src/            # Core model + data + graph logic
├── experiments/    # Training, ablation, cold-start
├── results/        # Evaluation outputs
├── demo.ipynb      # End-to-end walkthrough
├── requirements.txt
```

---

## 📂 Dataset

MovieLens-1M dataset:
https://grouplens.org/datasets/movielens/

---

## 🧠 Key Insight

Combining:

* Graph-based collaborative filtering
* Semantic embeddings
* LLM interaction

👉 leads to **more robust and practical recommender systems**

---

## 🔮 Future Work

* Full-scale hybrid evaluation
* Contrastive learning (SimGCL, SGL)
* Better LLM ranking integration

---

## 📌 References

* He et al., LightGCN (SIGIR 2020)
* Reimers & Gurevych, SBERT (EMNLP 2019)
* Rendle et al., BPR (UAI 2009)

---

## ⭐ Final Takeaway

> **LightGCN decides, SBERT improves, and LLM explains.**

