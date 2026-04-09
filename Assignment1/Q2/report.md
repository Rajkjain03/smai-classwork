# Question 2 — Data Normalisation and KNN Classification

## 2.1 Scenario Description

Roll Number: **2025201036**

Scenario ID:

Scenario ID = Roll Number mod 8

2025201036 % 8 = **4**

The dataset used in this experiment was collected dynamically using the provided API endpoint. Each data point was retrieved sequentially by polling the API until the server returned a 404 response indicating that no more samples were available.

The collected dataset contains **2001 samples** with **5 numerical features** and **1 class label**. Since each student receives a dataset generated based on their roll number and scenario ID, the classification problem differs slightly for every student.

---

## 2.2 Experimental Setup

The dataset was randomly shuffled using a fixed random seed of **42** to ensure reproducibility. It was then split into **80% training data and 20% testing data**.

All normalisation statistics (such as mean, standard deviation, minimum, maximum, and quantiles) were computed **only using the training set**, following the correct fit–transform protocol. The computed statistics were then applied to both the training and testing datasets.

A **K-Nearest Neighbour (KNN)** classifier was used with **Euclidean distance** as the similarity metric.

For each of the **21 configurations (Raw data + 20 normalization methods)**, hyperparameter tuning was performed using **5-fold cross-validation** on the training set. The value of **k** was selected from the range:

k ∈ {1, 2, ..., 30}

The optimal k obtained from cross-validation was used to train the final model on the full training set. The trained model was then evaluated on the test set using:

- **Test Accuracy**
- **Macro F1-score**

---

## 2.3 Master Results Table

The following table summarises the performance of all normalization methods. Each row corresponds to one preprocessing method applied before training the KNN classifier.

| Method | Best k | Test Accuracy | Macro F1 | Rank |
|------|------|------|------|------|
| RankGauss | ... | ... | ... | 1 |
| ZScore | ... | ... | ... | 2 |
| MinMax | ... | ... | ... | 3 |
| Raw | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| Softmax | ... | ... | ... | 21 |

The methods were ranked according to **test accuracy**, with rank 1 indicating the best performing normalization technique.

The results indicate that **RankGauss normalization produced the highest classification accuracy**, while **Softmax scaling performed significantly worse** than the other methods.

This suggests that certain transformations can improve distance-based models like KNN by standardising the distribution of feature values.

---

## 2.4 Visual Analysis

### 2.4.1 Feature Distribution

To understand the effect of normalization, a histogram and KDE plot were generated for one feature using three different preprocessing methods:

- Raw data
- Best performing normalization method (RankGauss)
- Worst performing normalization method (Softmax)

![Feature Distribution](feature_distribution.png)

From the plots, it can be observed that:

- The **raw data distribution** is approximately symmetric but slightly spread out.
- **RankGauss normalization** transforms the data into a distribution that closely resembles a Gaussian distribution.
- **Softmax normalization** compresses most values near zero and introduces heavy skewness.

Since KNN relies on Euclidean distances, this skewness can distort the relative distances between samples and negatively affect classification performance.

---

### 2.4.2 Accuracy vs k

The following plot shows how classification accuracy varies with different values of k for three configurations:

- Raw data
- Best performing normalization (RankGauss)
- Worst performing normalization (Softmax)

![Accuracy vs k](accuracy_vs_k.png)

From the plot we observe:

- **Raw data and RankGauss normalization achieve consistently high accuracy across different values of k.**
- **Softmax normalization produces significantly lower accuracy**, remaining around 0.45–0.50 across all k values.
- The results demonstrate that **appropriate feature scaling can significantly impact the performance of distance-based classifiers such as KNN.**

---

## 2.5 Discussion

Distance-based machine learning algorithms such as KNN are highly sensitive to the scale and distribution of input features. Normalization ensures that all features contribute equally to distance calculations.

In this experiment, **RankGauss normalization produced the best performance**, likely because it transforms the feature distribution into an approximately Gaussian form, reducing the influence of extreme values and improving distance consistency.

On the other hand, **Softmax scaling performed poorly**, as it compresses feature values into a narrow range and distorts the relative distances between samples.

Overall, the experiment demonstrates the importance of appropriate preprocessing when applying KNN classifiers to real-world datasets.
