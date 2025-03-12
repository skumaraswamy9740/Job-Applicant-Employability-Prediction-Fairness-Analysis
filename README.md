# cse190-assignment2

Link to data:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hopitals+for+years+1999-2008

Paper for where it came from:
https://onlinelibrary.wiley.com/doi/10.1155/2014/781670

More info on dataset:
https://fairlearn.org/main/user_guide/datasets/diabetes_hospital_data.html

Research Paper for Adverisial Neural Network Fairness Intervention:
https://www.researchgate.net/publication/341523922_Achieving_Fairness_with_Decision_Trees_An_Adversarial_Approach <br>
Achieving Fairness with Decision Trees: An Adversarial Approach (Vincent Grari1  · Boris Ruf2 · Sylvain Lamprier1 · Marcin Detyniecki2)

# Job Applicant Employability Prediction & Fairness Analysis  

## Overview  
This project develops a **machine learning pipeline for predicting job applicant employability** while evaluating and mitigating **bias across demographic attributes**. We employ a combination of **pre-processing, in-processing, and post-processing interventions** to improve fairness while maintaining model accuracy.  

As part of our **in-processing techniques**, we implemented an **adversarial debiasing approach** inspired by the research paper **"Achieving Fairness with Decision Trees: An Adversarial Approach"**. This method integrates a **neural network adversary** to minimize demographic influence in predictions. However, due to the **multiclass nature** of our sensitive attributes (**Man, Woman, NonBinary**), additional interventions such as **instance reweighting and separate group models** were also applied to refine fairness across groups.  

By combining these strategies, we **significantly reduced fairness gaps**, with **post-processing threshold tuning proving to be the most effective**, achieving nearly identical **equal opportunity values** across all groups.  

---

## Methodology & Process  

### 1. Data Processing & Feature Engineering  
- **Dataset Acquisition & Cleaning:**  
  - Sourced applicant data from **job portals, career fairs, and online applications**.  
  - Addressed **missing values, outliers, and categorical encoding** for ML compatibility.  
- **Feature Engineering:**  
  - **Created skill count features** to assess applicant expertise.  
  - Encoded demographic variables (**Gender, Age, Education Level**) to analyze fairness impact.  

### 2. Model Training & Hyperparameter Optimization  
- Trained a **Random Forest Classifier**, using **RandomizedSearchCV** for hyperparameter tuning.  
- Evaluated model performance with **accuracy, precision, recall, F1-score, and mean squared error (MSE)**.  

### 3. Fairness Assessment & Bias Detection  
- Implemented **demographic parity, equal opportunity, and predictive parity metrics**.  
- Analyzed **p% rule violations** to identify disparities across gender, age, and education groups.  
- Evaluated **false positive & false negative rates** for different demographic segments.  

### 4. Fairness Interventions & Results  

#### **Pre-Processing (Dataset Adjustments)**  
- Applied a **massaging approach**, promoting/demoting samples near the decision boundary.  
- **Results:** Reduced the maximum **equal opportunity gap from 10.8% to 6%**, nearly a **2x improvement**, while maintaining the F1 score at **0.805**.  

#### **In-Processing (Model-Level Adjustments)**  
- **Instance Reweighting** (Kamiran & Calders’ approach): Assigned sample weights dynamically based on demographic distribution.  
  - **Results:** Reduced the **man-woman gap from 5.5% to 3%**, but was less effective for **woman-nonbinary (10.8% gap remained high)**.  
  - F1 score remained **0.7930**.  

- **Separate Group Models (Majority Voting Approach)**: Trained separate models for **each gender group** and used **majority voting** for final predictions.  
  - **Results:**  
    - **Man-Woman gap reduced from 5.5% to 2.6%**  
    - **Woman-Nonbinary gap reduced from 10.8% to 3.3%**  
    - **F1 score slightly improved to 0.8095** due to the ensemble effect.  

#### **Post-Processing (Threshold Tuning) - Most Effective**  
- Adjusted **classification thresholds per group** to balance positive prediction rates.  
- **Results:**  
  - **Equal opportunity values became nearly identical** across all gender groups.  
  - Largest remaining gap was just **0.01%**.  
  - **F1 score remained stable at 0.8054**, avoiding significant accuracy loss.  

### 5. Adversarial Debiasing Approach  
- Inspired by **"Achieving Fairness with Decision Trees: An Adversarial Approach"**, we implemented a **min-max adversarial neural network** using **PyTorch**.  
- **How it works:**  
  - A **predictor model** minimizes classification error.  
  - An **adversarial network** attempts to **recover sensitive attributes** from predictions.  
  - The predictor **learns to "hide" demographic influence**, reducing bias.  
- **Results:**  
  - **Fairness improved, but not as effectively as expected** due to the **complexity of multiclass sensitive attributes**.  
  - **Demographic parity improved, but equal opportunity was harder to balance**.  
  - Trade-off required between **fairness and accuracy**.

---

## Results Summary  
| **Method**               | **Fairness Gap Before** | **Fairness Gap After** | **F1 Score**  |  
|-------------------------|----------------------|---------------------|--------------|  
| Baseline Model          | **10.8%**             | N/A                 | **0.8048**    |  
| Pre-Processing (Massaging) | **10.8%**             | **6%**               | **0.8050**    |  
| In-Processing (Reweighting) | **10.8%**             | **3%**               | **0.7930**    |  
| In-Processing (Separate Models) | **10.8%**             | **2.6%**             | **0.8095**    |  
| Post-Processing (Threshold Tuning) | **10.8%**             | **0.01%**            | **0.8054**    |  
| Adversarial Debiasing   | **10.8%**             | **~1.5%**            | **0.8010**    |  

---

---

## Key Technologies & Skills  
✔ **Machine Learning:** Random Forest, Logistic Regression, Model Optimization  
✔ **Fairness & Bias Detection:** Demographic Parity, Equal Opportunity, p% Rule  
✔ **Adversarial ML:** Min-Max Optimization, Fairness-Conscious Neural Networks (PyTorch)  
✔ **Hyperparameter Tuning:** RandomizedSearchCV for Model Optimization  
✔ **Feature Engineering & Data Processing:** Handling Bias in Structured Data  
✔ **Scalability & Implementation:** Python (Scikit-learn, Pandas, NumPy, PyTorch)  

---
