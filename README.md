# Income Prediction Project

## Project Description

This project aims to predict an individual's income level (whether it's `>50K` or `<=50K` per year) based on various demographic and socioeconomic factors. The process involves data cleaning, exploratory data analysis (EDA) to uncover key trends, and building a machine learning model to make predictions. The insights derived from this analysis are presented through clear, non-technical visualizations to inform stakeholders.


## Features

The dataset includes features such as:
*   **age**: Age of the individual.
*   **workclass**: Type of employer (e.g., Private, Self-emp-not-inc).
*   **fnlwgt**: Final weight (statistical weight).
*   **education**: Highest level of education achieved.
*   **educational-num**: Numeric representation of education level.
*   **marital-status**: Marital status (e.g., Married-civ-spouse, Never-married).
*   **occupation**: Type of occupation.
*   **relationship**: Relationship status (e.g., Husband, Not-in-family).
*   **race**: Race of the individual.
*   **gender**: Gender of the individual.
*   **capital-gain**: Capital gains.
*   **capital-loss**: Capital losses.
*   **hours-per-week**: Hours worked per week.
*   **native-country**: Country of origin.
*   **income**: Target variable (whether income is `>50K` or `<=50K`).

## Data Preprocessing

The following steps were taken to clean and prepare the data:
*   **Duplicate Handling**: Identified and removed duplicate rows.
*   **Missing Values**: Replaced '?' characters, which represented missing values, with 'Unknown' in categorical columns (`workclass`, `occupation`, `native-country`) to avoid introducing bias and allow the model to learn from their absence.
*   **Feature Engineering for Capital**: Transformed `capital-gain` and `capital-loss` into binary features (`Has_Capital_Gain`, `Has_Capital_Loss`) to indicate the presence or absence of capital transactions, and then dropped the original columns. This was done after observing that most entries for these columns were zero.

## Exploratory Data Analysis (EDA)

EDA focused on understanding the distribution of income across key demographic variables. Two key visualizations were created to highlight important trends for a non-technical audience:

### 1. Income Level Counts by Marital Status

This visualization shows how income levels are distributed across different marital statuses.
*   **Insight**: Individuals who are 'Married-civ-spouse' show a significantly higher proportion of earning more than 50K compared to other marital statuses. Conversely, groups like 'Never-married', 'Divorced', and 'Separated' predominantly fall into the `<=50K` income bracket. This trend could be influenced by factors such as shared financial resources, dual-income households, and career stability often associated with civil marriages.

### 2. Income Level Counts by Education Level (educational-num)

This plot illustrates the relationship between educational attainment and income.
*   **Insight**: There is a clear and strong positive correlation between higher educational attainment (represented by `educational-num`) and the likelihood of earning `>50K`. As the `educational-num` increases, the proportion of individuals with an income greater than 50K consistently rises, highlighting education as a critical factor for economic advancement.

## Modeling

### Approach:
1.  **Feature Selection and Target Definition**: `income` was set as the target variable, converted into a binary (0/1) format.
2.  **Data Splitting**: The dataset was split into training and testing sets.
3.  **Preprocessing Pipeline**: A `ColumnTransformer` was used to apply:
    *   `StandardScaler` to numerical features (`age`, `fnlwgt`, `educational-num`, `hours-per-week`, `Has_Capital_Gain`, `Has_Capital_Loss`).
    *   `OneHotEncoder` to categorical features (`workclass`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `native-country`).
4.  **Handling Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to address class imbalance, ensuring the model is not biased towards the majority class.
5.  **Model Training**: A `RandomForestClassifier` was trained on the preprocessed and balanced training data.
6.  **Evaluation**: The model's performance was evaluated using classification reports and confusion matrices on both training and test sets.

### Model Performance (Test Data):
*   **Accuracy**: Approximately 83%
*   **Precision (for >50K)**: Approximately 65%
*   **Recall (for >50K)**: Approximately 66%
*   **F1-Score (for >50K)**: Approximately 66%

## Key Insights

*   **Education and Marital Status are Strong Predictors**: Our exploratory analysis and model features show that higher education and being in a 'Married-civ-spouse' marital status are strongly associated with higher income.
   <img width="1189" height="690" alt="download" src="https://github.com/user-attachments/assets/bbf0a247-3faa-465c-864a-f8902aaf0be3" />
   
   <img width="1189" height="690" alt="download" src="https://github.com/user-attachments/assets/cd9ad687-fe57-4075-9144-3709284da354" />


*   **Impact of Hours Worked**: While not visualized in the final simplified plots, initial analysis (and permutation importance) also suggested that hours worked per week play a significant role in income levels, with those working more hours having a higher likelihood of earning >50K.
*   **Feature Importance**: Permutation importance highlighted `educational-num`, `age`, `hours-per-week`, `fnlwgt`, `Has_Capital_Gain`, `Exec-managerial` occupation, and `Married-civ-spouse` marital status as the most influential features for income prediction.
