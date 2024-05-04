# Credit Card Approval Prediction

## Introduction
This project aims to predict credit card approval using machine learning models based on various factors such as income, employment status, education level, and more. Credit card approval is a critical task for banks, and accurately predicting approval can help minimize risks and improve profitability.

## Dataset
The dataset used for this project contains information about clients, including their gender, car ownership, property ownership, annual income, type of income, education level, marital status, housing type, age, employment experience, and more. The dataset also includes a label indicating whether the credit card application was approved or rejected.
Here's the updated description of all the columns in the credit card dataset:

### Column Names and Descriptions

| Column Name       | Description                                                                                                      |
|-------------------|------------------------------------------------------------------------------------------------------------------|
| Ind_ID            | Client ID - Unique identifier for each individual                                                                |
| Gender            | Gender information of the individual                                                                             |
| Car_owner         | Indicates whether the individual owns a car (Yes/No)                                                             |
| Propert_owner     | Indicates whether the individual owns property (Yes/No)                                                          |
| Children          | Count of children                                                                                                |
| Annual_income     | Annual income of the individual                                                                                  |
| Type_Income       | Type of income (e.g., salary, business, investment)                                                              |
| Education         | Education level of the individual                                                                                |
| Marital_status    | Marital status of the individual (e.g., single, married, divorced)                                               |
| Housing_type      | Living style of the individual (e.g., rent, own house, with parents)                                             |
| Birthday_count    | Use backward count from current day (0), -1 means yesterday                                                      |
| Employed_days     | Start date of employment. Use backward count from current day (0). Positive value indicates current unemployment |
| Mobile_phone      | Indicates whether the individual has any mobile phone (Yes/No)                                                   |
| Work_phone        | Indicates whether the individual has any work phone (Yes/No)                                                     |
| Phone             | Indicates whether the individual has any phone number (Yes/No)                                                   |
| EMAIL_ID          | Indicates whether the individual has any email ID (Yes/No)                                                       |
| Type_Occupation   | Occupation of the individual                                                                                     |
| Family_Members    | Family size                                                                                                      |
| Label             | Application approval status: 0 for approved, 1 for rejected                                                      |



## Objective
The primary objective of this project is to analyze the impact of different factors on credit card approval rates and build predictive models to improve the accuracy of credit card approval predictions.

## Approach
### Data Analysis Track
- Hypothesis Testing: To assess relationships between factors such as income and credit approval.
- Feature Engineering: Creating new features and transforming existing ones to enhance model performance.
- Data Visualization: Visualizing data distributions and relationships between variables.

### Machine Learning Track
- Model Building: Training and evaluating machine learning models such as Logistic Regression, Random Forest, XGBoost, and SVM.
- Performance Evaluation: Assessing model performance using metrics like accuracy, precision, recall, and F1 score.
- Cross-validation: Evaluating model stability and generalization using cross-validation techniques.

## Results
- The Random Forest and XGBoost classifiers exhibited the highest accuracy and performance metrics compared to Logistic Regression and SVM.
- Feature engineering and oversampling techniques helped improve model performance and address class imbalance issues.
- Cross-validation results confirmed the stability and generalization ability of Random Forest and XGBoost models.

## SQL Queries
The project also includes SQL queries to analyze the dataset and answer specific questions about credit card approval, customer demographics, income distribution, and more.


## Dependencies
- Python 3.x
- Jupyter Notebook
- NumPy
- pandas
- seaborn
- scikit-learn
- XGBoost
- Random forest
- duckdb

  **1. Why is your proposal important in today’s world? How predicting a good client is worthy for a bank?**
  
Ans: Our proposal is important today because credit cards are used a lot as a financial tool. With more online shopping and transactions, credit cards are a popular way to pay. Banks need to quickly and 
     accurately process credit card applications. Predicting a good client is important for a bank because it helps reduce the risk of not getting paid back and makes managing credit card accounts cheaper. Banks 
    can use machine learning to analyze things like income and job status to predict which clients are less risky. This can lead to better profits for the bank and happier customers.

**2. How is it going to impact the banking sector?**

Ans: Impact of machine learning models to predict credit card approval in the banking sector:

1. *Improved efficiency:* Machine learning models can process large amounts of data quickly and accurately, allowing banks to process credit card applications more efficiently.

2. *Enhanced decision-making:* Machine learning models can analyze a wide range of factors and identify the most important ones that influence credit card approval, helping banks to make more informed and 
    accurate decisions.

3. *Improved profitability:* By accurately predicting credit card approval, banks can offer credit cards to clients with lower risk profiles, which can ultimately lead to better profitability.

4. *Enhanced customer satisfaction*: By offering credit cards to clients with lower risk profiles, banks can improve customer satisfaction and encourage repeat business.

**3. If any, what is the gap in the knowledge or how your proposed method can be helpful if required in future for any bank in India.**

Ans: The proposed method addresses a knowledge gap in credit card approval processes in Indian banks. Traditional methods may fall short in accurately assessing creditworthiness. Our machine learning approach 
     offers a data-driven solution to evaluate factors like income and employment status, enhancing approval predictions. In the future, this method can help Indian banks make more informed decisions, adapt to 
     changing economic landscapes, and minimize credit default risks, ultimately leading to better credit management.However, to fully leverage the benefits of our proposed method, banks will need to ensure that 
     they have access to reliable and comprehensive data sources.
