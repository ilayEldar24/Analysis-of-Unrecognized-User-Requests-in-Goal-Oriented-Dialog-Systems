# Analysis of Unrecognized User Requests in Goal-Oriented Dialog Systems

## Overview
This project focuses on analyzing **unrecognized user requests** in goal-oriented dialog systems (e.g., virtual assistants). The goal is to efficiently process large-scale unhandled requests by:
1. **Clustering** similar requests into meaningful groups.
2. **Extracting representative requests** from each cluster.
3. **Assigning meaningful names** to the generated clusters.

The solution employs **unsupervised learning**, **sentence embeddings**, and **clustering algorithms** to automate this process.

---

## Features
- ✅ **Dynamic Clustering Algorithm**  
  - Uses **SentenceTransformer** embeddings for text representation.  
  - Implements **centroid-based clustering** with **adaptive assignment**.  
  - Supports **iterative refinement** for better clustering.  

- ✅ **Representative Selection**  
  - Selects **diverse examples** from each cluster using **K-Means** within clusters.  

- ✅ **Cluster Naming**  
  - Uses **TF-IDF keyword extraction** for meaningful naming.  
  - Falls back to **noun phrase extraction** when necessary.  

- ✅ **Evaluation Pipeline**  
  - Compares clustering results against a **ground truth** using `evaluate_clustering.py`.  
  - Produces structured **JSON output**.  

---

## Installation

### 1. Install Dependencies
Ensure you have Python installed. Then, run:
```bash
pip install -r requirements.txt
```

### 2. Requirements
The following libraries are required:
```txt
numpy
pandas
scikit-learn
spacy
sentence-transformers
nltk
```
To install manually, use:
```bash
pip install numpy pandas scikit-learn spacy sentence-transformers nltk
```
For **spaCy**, download the required language model:
```bash
python -m spacy download en_core_web_sm
```

---

## Usage

### 1. Prepare Input Data
Ensure your input is a **CSV file** with a column named `"text"`, containing unrecognized user requests:
```csv
text
"how to reset my password?"
"what is the difference between covid and flu?"
"how to open a new account?"
...
```
Update the **config.json** file with the dataset path.

### 2. Run the Clustering Pipeline
To run the clustering and analysis, execute:
```bash
python main.py
```
This will:
1. **Load and preprocess** the dataset.
2. **Generate embeddings** for each request.
3. **Perform clustering** based on **semantic similarity**.
4. **Filter out small clusters**.
5. **Extract representatives** from each cluster.
6. **Assign meaningful names** to clusters.
7. **Save results** in structured **JSON format**.

### 3. Evaluate Clustering Performance
To compare the generated clusters against a **ground truth**, run:
```bash
python -m compare_clustering_solutions example_solution.json output.json
```

---

## Output Format
The results are saved in **output.json**, structured as follows:
```json
{
    "cluster_list": [
        {
            "cluster_name": "password reset",
            "requests": [
                "how to reset my password?",
                "forgot password help",
                "reset my login credentials"
            ],
            "representatives": [
                "how to reset my password?"
            ]
        }
    ],
    "unclustered": [
        "random outlier request that didn't fit into any cluster"
    ]
}
```

---

## Code Structure
```
📂 project_root
│── main.py                      # Entry point for clustering and evaluation
│── config.json                   # Configuration file with dataset paths
│── dynamic_clustering.py         # Clustering logic
│── cluster_namer.py              # Cluster naming logic
│── compare_clustering_solutions.py # Evaluation script
│── requirements.txt              # Required dependencies
│── output/                       # Directory for generated output
```

---

## Classes Overview

### 1. `DynamicClustering` (Core Clustering Algorithm)
Clusters unrecognized requests using **semantic similarity**.

#### Usage
```python
clustering_model = DynamicClustering()
embeddings = clustering_model.generate_embeddings(requests)
clusters, centroids, unclustered = clustering_model.assign_to_clusters(embeddings)
clusters, centroids, unclustered = clustering_model.refine_clusters(embeddings, clusters, centroids)
valid_clusters, final_unclustered = clustering_model.filter_clusters(clusters, unclustered)
reps = clustering_model.extract_representatives(requests, embeddings, valid_clusters)
```

---

### 2. `ClusterNamer` (Cluster Naming)
Assigns meaningful names to clusters.

#### Usage
```python
namer = ClusterNamer()
cluster_name = namer.extract_cluster_name(["how to reset my password?", "reset password help"])
print(cluster_name)  # → "Password Reset"
```

---

## Full Workflow Example
```python
clustering_model = DynamicClustering()
namer = ClusterNamer()

requests = ["how to reset my password?", "difference between covid and flu?", ...]

# Step 1: Generate embeddings
embeddings = clustering_model.generate_embeddings(requests)

# Step 2: Initial clustering
clusters, centroids, unclustered = clustering_model.assign_to_clusters(embeddings)

# Step 3: Refinement
clusters, centroids, unclustered = clustering_model.refine_clusters(embeddings, clusters, centroids)

# Step 4: Filter small clusters
valid_clusters, final_unclustered = clustering_model.filter_clusters(clusters, unclustered)

# Step 5: Extract representatives
reps = clustering_model.extract_representatives(requests, embeddings, valid_clusters)

# Step 6: Generate cluster names
cluster_names = [namer.extract_cluster_name([requests[idx] for idx in cluster]) for cluster in valid_clusters]

# Step 7: Save results
save_clustered_requests("output.json", valid_clusters, final_unclustered, reps, cluster_names, requests)
```

---

## Customization
You can modify:
- **Clustering parameters** (`similarity_threshold`, `min_cluster_size`, etc.).
- **Cluster naming logic** (e.g., increasing `top_n` keywords).
- **Sentence embedding model** (e.g., `"paraphrase-MiniLM-L6-v2"`).

---

## Acknowledgments
This project was developed as part of an **NLP Final Project (Jan 2025)** following best practices in **unsupervised clustering, text embeddings, and clustering evaluation**.
