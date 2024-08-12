# Team Project 1 - Group 9

## Description

This project analyzes data set obtained from patients diagnosed with Chronic Kidney Disease (CKD). The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, medication usage, symptoms, quality of life scores, environmental exposures, and health behaviors. 

Source: https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis

## Introduction

Chronic kidney disease (CKD) is a dangerous disease that can last a person’s entire life and is caused by either kidney malignancy or it decreased kidney functioning. CKD is a condition in which the kidneys are so damaged that they can’t filter blood effectively. The kidneys’ main job is to remove waste and excess water from the blood. CKD means that waste accumulates in the body. It is called chronic because the damage happens slowly over a long period of time. CKD affects people worldwide, and conditions such as diabetes, high blood pressure, and heart disease can lead to it. Age and gender also influence the likelihood of developing CKD. The two most common illnesses that might cause long-term damage to the kidneys are diabetes and high blood pressure, so controlling these diseases is crucial for CKD prevention. Many people with CKD do not realize they have it until it is too late, as symptoms often do not appear until the disease is advanced.

Early detection can halt or slow the progression of CKD, potentially preventing the need for dialysis or a kidney transplant. Identifying certain indicators in the dataset that predict CKD and emphasizing the importance of regular testing can lead to better outcomes and save lives.

Since the dataset contains demographic information, lifestyle factors, health behaviours, environment and occupational exposures, the data analysis could potentially analyze disparities in CKD prevalence in different groups. It could potentially highlight inequalities in healthcare access. This analysis can help patients by providing them with better information so that they can make better decisions about their care. 

This dataset contains detailed health information for 1,659 patients. Each patient is uniquely identified by a Patient ID. It also includes a confidential column indicating the doctor in charge. 

This project investigates the potential of using different machine learning models for providing an early diagnosis of CKD. In a supervised learning environment, four machine learning algorithms are employed to predict CKD accurately.

For a detailed description of the predictors or features in this dataset and their meanings,  please refer to the Data Card section of the Kaggle link [(here)(https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis)].

## Team members

1. Kateryna Skoropad
2. Anjali Deshpande 
3. Zarrin Rasizadeh
4. Adithya Hadadi


## Project Approach

### Phase 1

The team conducted a brainstorming session using the extensive list in [Team Project Part 1](https://github.com/UofT-DSI/team_project/blob/main/team_project_1.md) provided to us. With the valuable expertise of two doctors in our team, we swiftly decided to focus on medical-related datasets. Our primary considerations centred around the Thyroid and Chronic Kidney Disease (CKD) datasets. Initially, we discussed potential questions that the CKD dataset could help us address. Recognizing the need for preliminary data analysis (EDA) to refine our questions, we explored how various features in the dataset contribute to predicting CKD or non-CKD cases. 

The outcomes of our brainstorming sessions were documented [here](https://github.com/adithya28/team_project/blob/team-project-1/supporting_documents/CKD-data-info-questions.md).

We initiated our work with an empty notebook, CKD.ipynb, initially containing basic metadata blocks. Each block was assigned to specific programmers, detailing tasks and responsibilities. Guided by the principles of Exploratory Data Analysis (EDA) outlined in [this article](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/), our objective was to apply the statistical learning concepts covered in the 'Applying Statistical Learning' module of our program. Four models were implemented namely  Logistic Regression, K-Nearest Neighbors (KNN), Linear Regression, and Lasso Regression.
 
To streamline our development process, each programmer created a new branch named feature-\<number\> (e.g., feature-1, feature-2) for their assigned tasks. Following completion of each task, a pull request (PR) was created for review. PRs were tagged with the usernames of the assigned reviewers. As part of our commitment to sharing responsibilities, each team member took turns creating, reviewing, and merging pull requests, ensuring that quality was maintained. Throughout the review process, programmers continued iterating on their respective feature branches to refine and enhance their code. When the specific functionality was reviewed and finished, changes were merged into team-project-1 branch to integrate the functionality.

### Phase 2

We started with the notebook file (main source code) from phase 1 of the project, using it as our baseline. The Phase 1 code was reviewed and iterated upon. We studied the section under 'Overall Insights and Areas of Improvement' in the [(Readme)](https://github.com/adithya28/team_project/blob/team-project-1/README-project-1.md) of the first phase. A comprehensive task list was created after discussion, which can be found 
[(here)(https://drive.google.com/file/d/18sClhYqB4m8HqLS3fzqJMZ6Mwyej4zJ_/view)].

There was a need to have a concise display of bar plots and frequency plots in the EDA section of the notebook. We discussed the need for pipelines that included pre-processing steps and classifiers, handling normalization, data balancing with SMOTE and feature engineering. The framework could contain functions that each team member could use for implementation and tuning of various models to make the code more modular and scalable. This approach would ensure fewer bugs and allow passing the same pre-processed data to all models.

Models had to be evaluated using cross-validation techniques, including K-Fold and Stratified K-Fold. There was also a need for hyperparameter tuning using GridSearchCV and Random Search. This functionality had to be built into the functions/framework to be used by each programmer.

We wanted to further evaluate existing models (Logistic Regression and KNN) and try tree-based models like Random Forest and Gradient Boosting Algorithms such as XGBoost. PCA for dimensionality reduction was also to be experimented with for some of the models.

Consistent visualization of results with all the tuned model was to be implemented. We created common functions for visualizating the results using various plots. 
These included ROC curves, confusion matrices, Precision-Recall curves, Probability Prediction plots for model performance, and functions to evaluate Log Loss. We also wanted to implement functions for Permutation Importance plot, Partial Dependence Plot, Feature Importance plot, SHAP and beeswarm plot for model feature importance.

To streamline our development process, we followed the same guidelines as in Phase 1. Each programmer created a new branch named project-2-feature-\<number\> (e.g., project-2-feature-1, project-2-feature-2) for their assigned tasks. Following the completion of each task, a pull request (PR) was created for review. PRs were tagged with the usernames of the assigned reviewers. Each team member took turns creating, reviewing, and merging pull requests, ensuring that quality was maintained. Programmers continued iterating on their respective feature branches to refine and enhance their code based on inputs from other team members. When the specific functionality was finished, changes were merged into the team-project-2 branch to integrate the functionality, thereby automatically closing the Pull Request.

## Outcomes

### EDA and Data Analysis

#### Data Overview and Cleaning:

The dataset consisted of demographic, clinical, and lifestyle data of patients diagnosed with CKD.
Features like 'DoctorInCharge' and 'PatientID' were non-informative and were dropped.
Missing values were handled appropriately, and numerical data was normalized and standardized.

#### Exploratory Data Analysis (EDA) and Visualizations

##### Univariate Analysis

    

##### Bivariate Analysis


#####  Observations and Outcomes

**Health Indicators:**

Age, systolic BP, HbA1c levels, ACR, BUN levels, GFR, serum creatinine, and protein in urine were identified as significant indicators associated with CKD. Significant correlations (positive) were found between SerumCreatinine levels and the presence of CKD. GFR showed a negative correlation with CKD, indicating lower GFR values were associated with a higher likelihood of CKD diagnosis.

**Distribution Insights:**

Gender balance, ethnic distribution, socioeconomic status, education levels, smoking habits, and family medical history provided insights into the demographics and lifestyle factors of the population.

### Models and Predictions

### 1. Logistic Regression Model

#### Logistic Regression Model Dashboard

| Confusion Matrix | Log Loss |
|------------------|----------|
| ![Confusion Matrix](images/plot_confusion_matrix.png) | ![Log Loss](images/print_log_loss.png) |

| Predicted Probabilities Histogram | ROC Curve |
|-----------------------------------|-----------|
| ![Predicted Probabilities Histogram](images/plot_predicted_probabilities_histogram.png) | ![ROC Curve](images/plot_roc_curve.png) |

| Precision-Recall Curve | Permutation Importance |
|------------------------|------------------------|
| ![Precision-Recall Curve](images/plot_precision_recall_curve.png) | ![Permutation Importance](images/plot_permutation_importance.png) |

| Partial Dependence Plots | Feature Importance |
|--------------------------|--------------------|
| ![Partial Dependence Plots](images/plot_partial_dependence_plots.png) | ![Feature Importance](images/plot_feature_importance.png) |

| SHAP Beeswarm |
|---------------|
| ![SHAP Beeswarm](images/plot_SHAP_beeswarm.png) |

### 2. K-Nearest Neighbors (KNN) Model

#### KNN Model Dashboard

| Confusion Matrix | Log Loss |
|------------------|----------|
| ![Confusion Matrix](images/plot_confusion_matrix.png) | ![Log Loss](images/print_log_loss.png) |

| Predicted Probabilities Histogram | ROC Curve |
|-----------------------------------|-----------|
| ![Predicted Probabilities Histogram](images/plot_predicted_probabilities_histogram.png) | ![ROC Curve](images/plot_roc_curve.png) |

| Precision-Recall Curve | Permutation Importance |
|------------------------|------------------------|
| ![Precision-Recall Curve](images/plot_precision_recall_curve.png) | ![Permutation Importance](images/plot_permutation_importance.png) |

| Partial Dependence Plots | Feature Importance |
|--------------------------|--------------------|
| ![Partial Dependence Plots](images/plot_partial_dependence_plots.png) | ![Feature Importance](images/plot_feature_importance.png) |

| SHAP Beeswarm |
|---------------|
| ![SHAP Beeswarm](images/plot_SHAP_beeswarm.png) |

### 3. XGBoost Model

#### XGBoost Model Dashboard

| Confusion Matrix | Log Loss |
|------------------|----------|
| ![Confusion Matrix](images/plot_confusion_matrix.png) | ![Log Loss](images/print_log_loss.png) |

| Predicted Probabilities Histogram | ROC Curve |
|-----------------------------------|-----------|
| ![Predicted Probabilities Histogram](images/plot_predicted_probabilities_histogram.png) | ![ROC Curve](images/plot_roc_curve.png) |

| Precision-Recall Curve | Permutation Importance |
|------------------------|------------------------|
| ![Precision-Recall Curve](images/plot_precision_recall_curve.png) | ![Permutation Importance](images/plot_permutation_importance.png) |

| Partial Dependence Plots | Feature Importance |
|--------------------------|--------------------|
| ![Partial Dependence Plots](images/plot_partial_dependence_plots.png) | ![Feature Importance](images/plot_feature_importance.png) |

| SHAP Beeswarm |
|---------------|
| ![SHAP Beeswarm](images/plot_SHAP_beeswarm.png) |

### 4. Random Forest Classifier Model

## Areas of improvement

1. **Explore Feature Combinations**
   - Combine features such as age and BMI; Fasting Blood Sugar and HbA1C together.
   - Research if SystolicBP and DiastolicBP could be effectively combined.

2. **Explore Other Models**
   - Increase the complexity of current models to improve the ability to learn characteristics of class 0.
   - Experiment with Convolutional Neural Networks (CNNs).

3. **Feature Reduction and Feature Selection**
   Apply feature selection techniques to reduce the number of features and enhance model performance. 
   - **Choose and Evaluate Features**: Select a subset of the most important features based on specific criteria and use a model to evaluate this subset. Compare the performance of models trained with different feature subsets to determine the best trade-off between performance and complexity.
   - **Feature Selection Methods**:
     - **Chi-Square Test**: Assess the independence between categorical features and the target variable.
     - **ANOVA (Analysis of Variance)**: Test for significant differences in means for numerical features.
     - **Recursive Feature Elimination (RFE)**: Recursively remove features and build models to evaluate their impact on performance. RFE can be computationally expensive with large feature sets.
   - **Dimensionality Reduction**: This project experimented with Principal Component Analysis (PCA) using the KNN model. Consider exploring Linear Discriminant Analysis (LDA) as well.

4. **Explore Drug Relations**
   Investigate the correlation between CKD development and specific classes of drugs, such as diuretics, antidiabetic drugs, and NSAIDs. Examine these drugs in detail rather than as broad categories.

5. **Experiment Tracking Tools**
   Use tools like MLflow, Sacred, or TensorBoard for managing and analyzing machine learning experiments.

### Deliverables

1. Notebook file with EDA and models: [CKD.ipynb](https://github.com/adithya28/team_project/blob/team-project-2/src/CKD.ipynb)

2. Supporting documention: [CKD-data-info-questions.md](https://github.com/adithya28/team_project/blob/team-project-2/supporting_documents/CKD-data-info-questions.md)

3. [README file](https://github.com/adithya28/team_project/blob/team-project-2/README.project-2.md)

4. Video presentation of each team member:

    * Kateryna Skoropad - [Add Link Here]
    * Anjali Deshpande - [Add Link Here]
    * Zarrin Rasizadeh - [Add Link Here]
    * Adithya Hadadi - [Add Link Here]

### Rules of Engagement

#### Primary Communication Channels: 

Slack was used for day-to-day communication, while Microsoft Teams and Zoom was utilized for scheduled meetings and screen sharing.

#### GitHub Workflow:

* Each team member worked on individual feature branches (e.g., project-2-feature-1, project-2-feature-2 etc) for developing code.

* No direct changes were to be made to the team-project-2 branch. 

* Pull requests (PRs) were created for each feature or significant update.

* PRs were reviewed by another team member before merging into the team-project-2 branch. The comment section of PR was used to add review comments to help both the reviewer and the coder understand the changes.

* Subsequent reviews of the same file were done on the same pull request. The comment section of pull request was used to communicate the desired changes. Re-review was performed based on these comments. 

* On approval by another team-member the PR was merged to team-project-2 branch.

* Changes in team-project-2 were not to be merged into team-project-1. The two branches were kept independent of each other. 

#### Conflict Resolution:

* Team members collaborated to resolve conflicts during code merging.

* Persistent issues were escalated to other team members who had IT experience for resolution.

### Challenges

During the execution of the team project, we encountered the following challenges:

* Varying levels of technical skill proficiency.
* Insufficient time.
* Lack of experience of majority of team members working in data analysis area.
* Resolving merge conflicts: Errors occurrences when combining team member's code.


### Lessons Learned

In teamwork, open communication, mutual respect, and assistance allow us to overcome any difficulties.


### Directory structure

```markdown
|-- data
|---- processed
|---- raw
|---- sql
|-- reports
|-- src 
    |-- CKD.ipynb
|-- supporting_documents
    | -- supporting_documents/CKD-data-info-questions.md
|-- README-project-1.md
|-- .gitignore
```


