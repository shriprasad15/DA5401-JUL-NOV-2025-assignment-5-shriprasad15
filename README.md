

# Visualizing Data Veracity Challenges in Multi-Label Classification
### DA5401 Assignment 5

This project explores the challenges of real-world machine learning, specifically focusing on data veracity issues in a multi-label classification context. Using the Yeast dataset, we employ non-linear dimensionality reduction techniques—**t-SNE** and **Isomap**—to visually inspect the data for noisy labels, outliers, and hard-to-learn samples. The final output is a Jupyter Notebook that documents the entire process, from data preprocessing to visualization and insightful analysis.

---

**Details**
- **Name**: `S SHRIPRASAD`
- **Roll No**: `DA25E054`

---


## 🎯 Project Objective

The mission of this project was to act as data cartographers for the well-known **Yeast Dataset**. While this dataset is a common benchmark, it is fraught with real-world data quality issues that can severely impact the performance of a machine learning model. Our objective was to move beyond simple performance metrics and create a visual story that exposes these hidden challenges.

Using advanced dimensionality reduction, we aimed to answer the following questions:
- Can we visually identify genes that might be **mislabeled** or have ambiguous functions?
- Can we spot **outlier** experiments whose gene expression profiles are radically different from the rest?
- Can we map out the "chaotic frontiers" of the data—regions where different functional categories overlap so much that they become **hard to learn** for any classifier?

Ultimately, the goal was not just to build a model, but to deeply understand the data's intrinsic structure and its inherent challenges, thereby providing crucial intelligence for any future classification effort.

## 🔬 The Dataset

The analysis is performed on the **Yeast Dataset**, a standard benchmark for multi-label classification, sourced from the **Mulan Repository**.

-   **Instances**: Each of the 2,417 data points represents a single yeast gene.
-   **Features**: There are **103 continuous features** for each gene, corresponding to its microarray expression levels under various experimental conditions.
-   **Labels**: The target is a set of **14 binary labels**, where each label indicates whether the gene belongs to a specific functional category (e.g., "Metabolism," "Energy," "Cell Cycle"). Since a gene can have multiple functions, an instance can have multiple positive labels, making it a multi-label problem.

## 🛠️ Methodology

Our approach was a systematic exploration, moving from data preparation to deep visual analysis. Each step was designed to build upon the last, creating a comprehensive picture of the data's landscape.

### 1. Preprocessing and Strategic Setup

Before visualization, the data required careful preparation to ensure our tools would work effectively and our insights would be meaningful.

-   **What we did:**
    1.  **Data Loading:** We loaded the feature matrix `X` (103 features) and the label matrix `Y` (14 labels) from the standard `.arff` file format using the `scipy` and `liac-arff` libraries.
    2.  **Label Simplification:** Plotting with 14 distinct labels would be visually overwhelming. We devised a simplification strategy to create a clear 4-color map. We identified and isolated the **two most frequent single-label classes** and the **single most frequent multi-label combination**, grouping everything else into an "Other" category. This allowed us to focus our visual analysis on the most dominant patterns in the data.
    3.  **Standardization:** We applied `StandardScaler` to the feature matrix.
-   **Why it was crucial:** Dimensionality reduction algorithms like t-SNE and Isomap are **distance-based**. Without scaling, features with large numerical ranges would dominate distance calculations, rendering features with smaller ranges irrelevant. Standardization places all features on an equal footing, ensuring an unbiased and accurate representation of the data's structure.

### 2. t-SNE for Local Structure Visualization

Our first tool, **t-SNE (t-Distributed Stochastic Neighbor Embedding)**, acts like a microscope, excels at revealing local neighborhoods and clustering tendencies.

-   **What we did:**
    1.  We experimented with the `perplexity` hyperparameter (5, 30, 50), settling on **30** as the optimal value that revealed clear, well-separated clusters without over-fragmenting the data.
    2.  We generated a 2D scatter plot of the yeast genes, colored by our simplified 4-category label system.
    3.  This plot served as our primary tool for visually identifying clusters, outliers, and regions of class overlap.
-   **Why we used it:** t-SNE's strength is its ability to group similar data points into tight visual clusters. This makes it exceptionally good at spotting anomalies—points that don't belong to their assigned cluster (potential mislabels) or points that form no cluster at all (outliers).

### 3. Isomap for Global Manifold Learning

Our second tool, **Isomap (Isometric Mapping)**, acts like a satellite, revealing the global, high-level structure of the data—what is known as the **data manifold**.

-   **What we did:**
    1.  We applied Isomap to the scaled data to generate a complementary 2D representation.
    2.  We compared its output to the t-SNE plot, observing that Isomap produced a more continuous, "unfurled" structure that captured the overarching shape of the data, whereas t-SNE emphasized cluster separation.
-   **Why we used it:** While t-SNE is excellent for local views, the distances between its clusters are not meaningful. Isomap preserves **geodesic distances** (paths along the curved data surface), giving us a more faithful representation of the data's global geometry. This helped us understand how the different clusters were fundamentally connected and assess the overall complexity of the classification problem.

### 4. Extra Analysis for Deeper Context

To add quantitative support to our visual findings, we conducted two additional analyses:
-   **Label Cardinality Plot:** Visualized the distribution of the number of labels per gene, confirming the dataset's strong multi-label nature.
-   **Label Co-occurrence Heatmap:** Revealed which functional categories frequently appear together, providing a biological rationale for why certain classes were intermingled in our visualizations.

## 📊 Key Findings

Our visual exploration successfully uncovered significant data veracity challenges that would pose a threat to any standard classification model.

1.  **Noisy/Ambiguous Labels Identified:** The t-SNE plot clearly showed numerous instances of one color deeply embedded within a large, dense cluster of another color. These points are "impostors" whose feature profiles strongly suggest they belong to a different class, indicating either a data entry error or a genuine biological ambiguity.
2.  **Outliers Detected:** We identified several isolated "lone wanderer" points, positioned far from any major cluster. These represent genes with highly anomalous expression profiles, which could be due to experimental error or represent rare, unique biological states.
3.  **Hard-to-Learn Regions Mapped:** Both visualizations, particularly t-SNE, revealed a "chaotic nebula" in the central region where points from all categories were thoroughly intermingled. This is a frontier of high uncertainty where class boundaries are inherently fuzzy, making it a minefield for any classifier.
4.  **Complex Data Manifold Confirmed:** The Isomap visualization showed that the data does not lie on a simple, flat plane but on a **complex, curved manifold**. This insight is critical, as it confirms that simple linear models are fundamentally unsuited for this problem and that more sophisticated, non-linear models are required to learn the curved decision boundaries.

## 💻 Tools Used
- **Language**: Python 3.x
- **Environment**: Jupyter Notebook / JupyterLab
- **Core Libraries**: Pandas, NumPy, Scikit-learn, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Data Loading**: `liac-arff`

## ⚙️ Setup and Installation

To run this analysis on your local machine, follow these steps.

**1. Clone the repository:**
```bash
git clone https://github.com/shriprasad15/DA5401-JUL-NOV-2025-assignment-5-shriprasad15.git
cd DA5401-JUL-NOV-2025-assignment-5-shriprasad15
```

**2. Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

**3. Install dependencies:**
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyterlab liac-arff
```

## ▶️ How to Run

1.  **Download the Dataset:** Download the `yeast.arff` file from the [MULAN Repository - Yeast Data](http://www.uco.es/kdis/mllresources/datasets/arff/yeast.arff) and place it in the root directory of this project.
2.  **Launch Jupyter:** Open a terminal in the project directory and run:
    ```bash
    jupyter lab
    ```
    or
    ```bash
    jupyter notebook
    ```
3.  **Open and Run the Notebook:** Open the `da25e054.ipynb` file and run the cells sequentially from top to bottom.

## 🗂️ File Structure
```
├── da25e054.ipynb    # The main Jupyter Notebook with all code and analysis.
├── yeast.arff                     # The dataset file (must be downloaded).
├── README.md                      # This file.
```
