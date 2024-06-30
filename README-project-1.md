# Team Project 1 - Group 9

## Description

This project analyzes data set obtained from patients diagnosed with Chronic Kidney Disease (CKD). The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, medication usage, symptoms, quality of life scores, environmental exposures, and health behaviors. 

Source: https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis

## Team members

1. Kateryna Skoropad
2. Anjali Deshpande 
3. Zarrin Rasizadeh
4. Adithya Hadadi

## Project Approach
	
The team conducted a brainstorming session using the extensive list in [Team Project Part 1](https://github.com/UofT-DSI/team_project/blob/main/team_project_1.md) provided to us. With the valuable expertise of two doctors in our team, we swiftly decided to focus on medical-related datasets. Our primary considerations centred around the Thyroid and Chronic Kidney Disease (CKD) datasets. Initially, we discussed potential questions that the CKD dataset could help us address. Recognizing the need for preliminary data analysis (EDA) to refine our questions, we embarked on exploring how various features in the dataset contribute to predicting CKD or non-CKD cases. 

The outcomes of our brainstorming sessions were documented [here](https://github.com/adithya28/team_project/blob/team-project-1/supporting_documents/CKD-data-info-questions.md).

We initiated our work with an empty notebook, CKD.ipynb, initially containing basic metadata blocks. Each block was assigned to specific programmers, detailing tasks and responsibilities. Guided by the principles of Exploratory Data Analysis (EDA) outlined in [this article](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/), our objective was to apply the statistical learning concepts covered in the 'Applying Statistical Learning' module of our program. 

Each metadata block outlined tasks and assigned responsibility to team members, facilitating effective task management and collaboration. These tasks were collectively agreed upon and divided among team members based on our areas of interest. This is how we split EDA tasks among team members. Then each of us decided to implement one regression model each and record our observations. The EDA tasks were also followed each by observations. 

To streamline our development process, each programmer created a new branch named feature-\<number\> (e.g., feature-1, feature-2) for their assigned tasks. Following completion of each task, a pull request (PR) was created for review. PRs were tagged with the username of the assigned reviewer. As part of our commitment to shared responsibility, each team member took turns creating, reviewing, and merging pull requests, ensuring comprehensive oversight and quality control. 

Each team member took turns creating, reviewing, and merging PRs, fostering shared responsibility and knowledge exchange. Throughout the review process, programmers continued iterating on their respective feature branches to refine and enhance their contributions.Upon approval of a PR related to specific functionality, changes were merged into team-project-1 branch to integrate new functionalities and maintain project coherence.

Post-coding deadline, our focus shifted towards producing a video presentation and preparing a detailed README file, summarizing our project's objectives, approach, methodologies, and outcomes.

## Learning Goals and Outcomes

### EDA and Data Analysis

#### Data Overview and Cleaning:

The dataset consisted of demographic, clinical, and lifestyle data of patients diagnosed with CKD.
Features like 'DoctorInCharge' and 'PatientID' were non-informative and were dropped.
Missing values were handled appropriately, and numerical data was normalized and standardized.

#### Exploratory Data Analysis (EDA) and Visualizations

##### Univariate Analysis

    1. Summary Statistics: Provided descriptive statistics including count, mean, standard deviation, and percentiles for numerical variables.

    2. Data Visualization: Utilized histograms to visualize distributions of numerical variables. For example, the 'Diagnosis' histogram showed a skew towards positive CKD diagnoses.

##### Bivariate Analysis

    1. Correlation Analysis:

        1. Pearson correlation coefficients were computed between numerical variables and the diagnosis ('Diagnosis') to identify relationships. GFR and SerumCreatinine showed higher co-relation with CKD.

        2. Visualized correlations using a heatmap, highlighting significant correlations such as SerumCreatinine positively correlating with Diagnosis and Glomerular Filtration Rate (GFR) negatively correlating.

    2. Relationships with CKD:

    Box Plots: Compared distributions of numerical variables between CKD and non-CKD groups. Observations included higher median age, higher systolic BP, poorer HbA1c control, and lower GFR among CKD patients.

#####  Observations and Outcomes

**Health Indicators:**

Age, systolic BP, HbA1c levels, ACR, BUN levels, GFR, serum creatinine, and protein in urine were identified as significant indicators associated with CKD. Significant correlations (positive) were found between SerumCreatinine levels and the presence of CKD. GFR showed a negative correlation with CKD, indicating lower GFR values were associated with a higher likelihood of CKD diagnosis.

**Distribution Insights:**

Gender balance, ethnic distribution, socioeconomic status, education levels, smoking habits, and family medical history provided insights into the demographics and lifestyle factors of the population.

### Models and Predictions


### 1. Logistic Regression Model

**Variables:** Numerous predictors including demographic, clinical measurements, lifestyle factors, and medical history.

**Preprocessing:** Standardization for numerical features, one-hot encoding for categorical features.

**Evaluation:** Achieved an accuracy of 93% on the test set.

**Precision and Recall:**

    1. Class 0 (CKD negative): Precision 1.00, Recall 0.08
    2. Class 1 (CKD positive): Precision 0.93, Recall 1.00

**ROC:**
Higher Area Under Curve (AUC) values indicates better model performance in distinguishing between CKD positive and negative cases.
There are steps in the ROC curve, as compared to a smooth curve, indicative of an imbalanced datasets, due to fewer changes in predicted probabilities for the minority class.

**Log loss:** The log loss value of 0.2 indicates high confidence and accuracy in the logistic regression model's predictions for CKD and non-CKD classes. This metric reinforces the model's effectiveness in probabilistic predictions.

**Confusion Matrix:** Highlights a high rate of correct predictions for CKD positive cases but struggles with CKD negative cases due to dataset imbalance.
Next Steps: Address dataset imbalance to improve model performance on CKD negative predictions.
  
### 2. K-Nearest Neighbors (KNN) Model

**Preprocessing:** Standardization of predictors.

**Evaluation:** Achieved an accuracy of 91.8% on the test set, but had challenges with sensitivity (recall) due to class imbalance in the dataset. 

**Precision and Recall:**

    1. Class 0 (no CKD): Precision 0.00, Recall 0.00
    2. Class 1 (CKD): Precision 0.92, Recall 0.99

Precision and F1-score were higher for CKD, indicating the model's ability to correctly classify CKD cases.

**Confusion Matrix:** Shows a strong ability to predict CKD (class 1) but struggles with class 0 due to severe class imbalance.

**Next Steps:** Addressing class imbalance and refining feature selection for improved performance.

### 3. Linear Regression Model

**Objective:** Predict Glomerular Filtration Rate (GFR) using linear regression with selected predictors.

**Predictors:** BUNLevels, SerumCreatinine, ProteinInUrine, HbA1c.

**Evaluation:** Poor fit indicated by low R-squared (0.003) and non-significant F-statistic (p = 0.309). The model exhibited low R-squared values, suggesting poor fit for GFR prediction, possibly due to non-linear relationships or missing influential predictors.

**Interpretation:** Variables do not explain GFR variability adequately, limiting predictive utility.

### 4. Lasso Regression Model

**Objective:** Use Lasso regression for feature selection and predicting CKD.

**Evaluation:**

    1. MSE: 0.082, indicating average squared difference between predicted and actual values.

    2. MAE: 0.155, showing average absolute difference.

    3. R-squared: -0.002, suggesting poor model fit due to inappropriate model choice or noisy data.

**Observations:** Model under-predicts CKD cases and indicates poor fit for predicting non-CKD status.


## Overall Insights and Areas of improvement

Logistic regression and KNN showed the best performance in predicting CKD, with logistic regression providing insights into influential predictors.

**Dataset Imbalance:** Class imbalance affected model performance, particularly in correctly identifying non-CKD cases. Almost all model predictions (logistic, KNN and Lasso) pointed towards the same imbalance. This resulting in models with high accuracy but biased towards the majority class (in our case, the diagnosis of CKD), and failing to capture the nuances in the minority class. Steps such as undersampling, oversampling, feature selection etc. should be taken to address this issue. Both logistic regression and KNN models faced challenges due to a significant imbalance between CKD positive and negative cases. Addressing this imbalance through techniques like oversampling or adjusting class weights could improve predictions for CKD negative cases. 

**Cross-Validation Strategies:** We might want want to use advanced cross-validation techniques (e.g., stratified K-fold) to ensure model performance stability.


**Oversampling:** Randomly replicate instances of the minority class (CKD cases) to balance class distribution (e.g., SMOTE - Synthetic Minority Over-sampling Technique). Generate synthetic data points for the minority class using techniques like SMOTE to balance the dataset. 

**Exploring Other models** Exploring tree-based algorithms is probably recommended, as they often perform well on imbalanced datasets. Additionally, boosting algorithms such as AdaBoost and XGBoost are ideal for such data because they give higher weight to the minority class at each successive iteration, adjusting the weights of misclassified classes during each training iteration.

**Exploring other relations**
In the CKD dataset, there is data on the use of diuretics, antidiabetic drugs, and NSAIDs. It would be interesting to investigate whether there is a correlation between the development of CKD and specific classes of these drugs, rather than just the groups as a whole.

### Deliverables

1. Notebook file with EDA and models: [CKD.ipynb](https://github.com/adithya28/team_project/blob/team-project-1/src/CKD.ipynb)

2. Supporting documention: [CKD-data-info-questions.md](https://github.com/adithya28/team_project/blob/team-project-1/supporting_documents/CKD-data-info-questions.md)

3. [README file](https://github.com/adithya28/team_project/blob/team-project-1/README.project-1.md)

4. Video presentation of each team member:

    * Kateryna Skoropad - https://drive.google.com/file/d/1TLpJWCkmaWgd1fHmtmGpn2MhnDYJ8i-S/view?usp=drive_link
    * Anjali Deshpande - https://drive.google.com/file/d/1sfN0kHJCwvecYuOK_Din1dbMg1AIh9EB/view?usp=sharing
    * Zarrin Rasizadeh - (Add link here)
    * Adithya Hadadi - https://drive.google.com/file/d/106kyntCuGyWkExNdhD3Esmu3zTc916c1/view?usp=sharing

### Rules of Engagement

#### Primary Communication Channels: 

Slack was used for day-to-day communication, while Microsoft Teams and Zoom was utilized for scheduled meetings and screen sharing.

#### GitHub Workflow:

* Each team member worked on individual feature branches (e.g., feature-1, feature-2 etc) for developing code.

* No direct changes were to be made to the team-project-1 branch. 

* Pull requests (PRs) were created for each feature or significant update.

* PRs were reviewed by another team member before merging into the team-project-1 branch.

* Subsequent reviews of the same file were done on the same pull request. The comment section of pull request was used to communicate the desired changes. Re-review was performed based on these comments. 

* On approval by another team-member the PR was merged to team-project-1 branch.


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


