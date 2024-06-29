# Team Project 1 - Group 9

## Description

This project analyzes data set obtained from patients diagnosed with Chronic Kidney Disease (CKD). The dataset includes demographic details, lifestyle factors, medical history, clinical measurements, medication usage, symptoms, quality of life scores, environmental exposures, and health behaviors. 

Source: https://www.kaggle.com/datasets/rabieelkharoua/chronic-kidney-disease-dataset-analysis

## Team members

1. Kateryna Skoropad
2. Anjali Deshpande 
3. Zarrin Rasizadeh
4. Adithya Hadidi

## Project Approach
	
The team conducted a brainstorming session using the extensive list provided in [Team Project Part 1](https://github.com/UofT-DSI/team_project/blob/main/team_project_1.md) file provided to us. With the valuable expertise of two doctors in our team, we swiftly decided to focus on medical-related datasets.Our primary considerations centred around the Thyroid and Chronic Kidney Disease (CKD) datasets. Initially, we discussed potential questions that the CKD dataset could help us address. Recognizing the need for preliminary data analysis (EDA) to refine our questions, we embarked on exploring how various features in the dataset contribute to predicting CKD or non-CKD cases. 

The outcomes of our brainstorming sessions were documented [here](https://github.com/adithya28/team_project/blob/team-project-1/supporting_documents/CKD-data-info-questions.md).

We initiated our work with an empty notebook, CKD.ipynb, initially containing basic metadata blocks. Each block was assigned to specific programmers, detailing tasks and responsibilities. Guided by the principles of Exploratory Data Analysis (EDA) outlined in [this article](https://www.geeksforgeeks.org/what-is-exploratory-data-analysis/), our objective was to apply the statistical learning concepts covered in the 'Applying Statistical Learning' module of our program. 

Each metadata block outlined tasks and assigned responsibility to team members, facilitating effective task management and collaboration. These tasks were collectively agreed upon and divided among team members based on our areas of interest. This is how we split EDA tasks among team members. Then each of us decided to implement one regression model each and record our observations. The EDA tasks were also followed each by observations. 

To streamline our development process, each programmer created a new branch named feature-\<number\> (e.g., feature-1, feature-2) for their assigned tasks. Following completion of each task, a pull request (PR) was created for review. PRs were tagged with the username of the assigned reviewer. As part of our commitment to shared responsibility, each team member took turns creating, reviewing, and merging pull requests, ensuring comprehensive oversight and quality control. 

Each team member took turns creating, reviewing, and merging PRs, fostering shared responsibility and knowledge exchange. Throughout the review process, programmers continued iterating on their respective feature branches to refine and enhance their contributions.Upon approval of a PR related to specific functionality, changes were merged into team-project-1 branch to integrate new functionalities and maintain project coherence.

Post-coding deadline, our focus shifted towards producing a video presentation and preparing a detailed README file, summarizing our project's objectives, approach, methodologies, and outcomes.

## Learning Outcomes

### Summary of analysis

#### EDA

#### Models

### Deliverables

1. Notebook file with EDA and models: [CKD.ipynb](https://github.com/adithya28/team_project/blob/team-project-1/src/CKD.ipynb)

2. Supporting documention: [CKD-data-info-questions.md](https://github.com/adithya28/team_project/blob/team-project-1/supporting_documents/CKD-data-info-questions.md)

3. [README file](https://github.com/adithya28/team_project/blob/team-project-1/README.project-1.md)

4. Video presentation of each team member:

* Kateryna Skoropad - (Add link here)
* Anjali Deshpande - (Add link here)
* Zarrin Rasizadeh - (Add link here)
* Adithya Hadidi - (Add link here)

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


