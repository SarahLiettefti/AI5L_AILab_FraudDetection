# AI5L Artificial Intelligence Report

## TODO
- [x] rajouter mes features sur ton script
- [x] runner un mi sur toutes les features including mine
- [x] evaluer avec les models de sarah. (fct)
- [x] Evaluer les données qu'on peut enlever:
    - [x] ex: batch id, account id etc
- [x] Tester le undersampling avec k-mean (sans hasard)
- [x] Tester méthode oversampling sans hasard. 
- [x] Intro
- [ ] spécificité du dataset (unbalanced)
- [ ] conséquence sur l'entrainement 
- [ ] Démarceh (comment a tester)
- [x] Feature enegineering
- [ ] Essaye sur plusieurs model et compare. Lesquel 
- [ ] Conclusion et perspective 
    - [ ] Rajouter des features : faire moyene de nombre de transaction par jour

Files to deal with later : 
- [ ] DataCleaningScript.py

L'évaluation des MI du feature  dans EvaluationFeatures.ipynb


# Introduction

In today's digital age, the rapid growth of online transactions has brought about an alarming increase in financial fraud. Detecting fraudulent transactions is crucial to maintain the security and trust of customers and financial institutions. The objective of this report is to provide an in-depth analysis and evaluation of machine learning models for the Xente Fraud Detection Challenge, a dataset containing transaction data from Xente, an e-commerce platform in Africa.

The challenge involves developing an effective fraud detection model that can identify potentially fraudulent transactions by analyzing and learning from historical transaction data. To accomplish this task, we will explore various aspects of the dataset, perform feature engineering to extract relevant information, and compare the performance of different machine learning models to identify the most suitable approach.

This report will present a detailed analysis of the Xente transaction dataset, including data preprocessing, feature engineering techniques employed, and the rationale behind each decision. Furthermore, we will evaluate the performance of our chosen model, comparing it with various other models to showcase its strengths and limitations. Finally, we will emphasize the importance of scientific reasoning and its role in enhancing the overall effectiveness of our fraud detection model.

By the end of this report, we aim to provide a comprehensive understanding of the steps involved in developing a robust fraud detection model, as well as insights into the performance and suitability of different machine learning techniques for this task. Our findings will contribute to the ongoing efforts to improve the security of online transactions and mitigate the risks associated with financial fraud.

Case of binary classification

## Specificity of the dataset
The training dataset is unbalanced, meaning that there is a minority class that is significantly  less represented in the dataset than the other class. Int this case, the minority class is the Fraud with only 193 data compared to 95469 non-fraud one. Although this is expected in real-life scenarios, it can lead to an overly sensitive model towards the majority class and produce biased and inaccurate predictions.

![](https://hackmd.io/_uploads/BJhusXBN3.png)

To address this issue, we have two main options:
1. Using a **model** that take into account the imbalanced data by assigning weights to each class. Setting a positive weight for a class increases the penalty for misclassifying instances of that class during training, making the model pay more attention to the minority class and potentially improving its ability to correctly classify instances of that class. Random Forest and XGBoost are examples of such models that will be used. The optimal positive weight is 22. It was calculates by the square root of the number of non-fraude in the dataset divided by the frauds. Several weights have been tested in the SecondModels.ipynb script, and this one had the best performances.
2. **Sampling** the dataset to balance the number of instances in each class. There are two types of sampling methods: **undersampling** and **oversampling**. **Undersampling** involves removing instances from the majority class, but this may not perform well when the remaining data is not sufficient for the model to learn from. In this case, it only left us with 386 instances which is not enough. The result from the website were around 0.007513148. **Oversampling**, on the other hand, involves creating new data for the minority class to balance the dataset. This had a better result (see later). We can sample randomly or with some intelligence behind it. Both case were tested. For the undersampling, we used k-means to create 193 clusters of the majority class and select 193 data from it. For the oversampling, the SMOTE (Synthetic Minority Oversampling Technique) method was chosen because it avoid overfitting by interpolating new instances between existing minority class instances. All these methods where tested in the TestEverything.ipynb script.

The result where balanced but the best were :
| Model       |Positive weight |Description|Public Score | Private Score |
| ----------- |-----------|-----------| ----------- |---- |
| XGBoost Classifier|22|Without the columns with very low MI|0.711864| 0.715447 
| Random Forest Classifier|None but used SMOTE to balance the data|36 decision three and the criterion = Entropy|0.709677|  0.686567
| Random Forest Classifier| 22|Basic data and 36 decision three and the criterion = Entropy| 0.703704       | 0.654867	
| Random Forest Classifier  |500| Basic data and 36 decision three and the criterion = Entropy| 0.678571       | 0.666667

We also got a 0.78 score (the website one) with very basic model and featuring engineering. But we couldn't reach this score again, even though we were doing everything to improve the model.

### How we evaluate
It is important to use the right metric to evaluate our models to have the right idea on its performance. In our case, we are doing a binary classification with unbalanced dataset. For that we are using five metrics (plus de mean of those). They are calculated and explain in the notebook EvaluationMetric.ipynb. 


We use those to have an idea of the performance of the models. We always split the data in training set and validation set before fitting it to the model to avoid target leakage.
1. Precision : $TP \over (TP+FP)$ It shows how sensitive is the model to the signal to be recognized. So how often we are correct when we classify a class positive. More it is close to 1 and better it is. A high-precision model, means that it doesn't always recognize a fraud, but when it predicts it,  it is certain of it. 
2. Recall : $TP \over (TP+FN)$ A model with a high recall will recognize as many positive classes as possible. We want a high recall if we want to be sure to detect all the fraud and don't care that sometimes it classifies non fraud as frauds. (Includes noises)
3. F1 score : combines precision and recall (2 complementary metrics) into one metric. $2 * {precision*recall} \over {precision+recall}$ It is probably the most used metric for evaluating binary classification models. If our F1 score increases, it means that our model has increased performance for accuracy, recall or both.
4. Log loss : This metric measures the difference between the probabilities of the model’s predictions and the probabilities of observed reality. The goal of this metric is to estimate the probability that an example has a positive class. More the log Loss is near 0 better it is.
5. Matthews Correlation Coefficient (MCC) : is designed to evaluate even models trained on unbalanced dataset. It ranges between $-1$ and $+1$. We want it to be near $+1$ since it indicates a correlation between actual observed values and predictions made by our model.

The mean of those values (except of the Log loss because it is close to 0) is used to just have a general idea of the performance.

We didn't use the accuracy metric, even if it is a common and simple metric it is not recommended for our case of unbalanced data. Because if the model only predicts 0, it can have an accuracy of 99,7% while having failed to predict all the frauds. 

Those metrics are used to have an idea of the performance. Once we are happy with the performance of a model, we use it to predict value from the test dataset (without the target) and evaluate it on the website. Usually the score on the website are around 0.66666 whereas for the same model with our metrics are between 0.8 and 0.98. This difference could be explained with the phenomena of overfitting. 


# Feature Engineering
## Preprocessing
We first need to set up a reference score with barely untouched data. Some processing might be still necessary in order to train a model.

As first easy step, we can counts the number of unique entries in each column of our training dataset, and then identifies columns with only one unique value. These columns can be dropped from both the training and testing datasets, as they do not provide any useful information for model training.

Next important step is to convert any object or string value into numerical value so we can train our model. We first find and process columns containing "Id" in their names as they contain string values. To make it trainable with our model we remove the column name prefix and converts the values in these columns to integers.

We can also obseve a timestamp for each transaction. As it is in string format we need to perform some conversion to make this data usable for the training process. So we convert the "TransactionStartTime" column to a datetime format, enabling the extraction of various time-based features. After some quick observation we see that transactions are spread across 4 month between 2018 and 2019. So keeping the month or the year is not usefull as we don't have any transaction for the other months or years. Still we can create new columns such as "TransactionDayOfWeek", "TransactionDayOfMonth", "TransactionHour", and "TransactionMinute". These features may help in capturing time-based patterns in the data. After the extraction, the original "TransactionStartTime" column must be dropped.

Following that, the "ProductCategory" column must be factorized as it contains string value for each category, which means the categorical data is encoded into numerical values. The factorized column is then converted to an integer data type for efficient processing.

Some model behave differently regarding discrete value (usually represented integer type) and continuous value (represented in float type). Therefore, we should convert "Value" column from integer to a float data type to avoid any problem later on for the training process. 

Finally, to avoid redundant information with Value column (absolute value of 'Amount' column), we can create a new binary column called "Expense", which indicates whether a transaction is an expense (Amount is negative) or not. The "Amount" column is then dropped, as its information is now captured by the "Expense" column.

## Base Score Evaluation

We can now evaluate on Xente website how our base preprocessed dataset performs with different classifier models such as RandomForestClassifier, TreeClasifier and XGBoostClassifier. We will keep training on those three models in this report in order to measure and compare performance with different dataset manipulation, part of the feature engineering process. 

This yield the following result which we can base our future training to improve our model :

| Model       | Public Score | Private Score | Dataset |
| ----------- | ----------- |---- |---|
| RandomForest | 0.666       | 0.649  | Stock
| TreeClasifier | 0.678       |  0.666 |Stock
| XGBClassifier      | 0.666       | 0.682  |Stock

After some investigation and using the feature description provided by Xente we see that some features have unique identifier. Those being unique may lead to inconclusive result when we're predicting completely new values which is the goal of the model. Therefore we can drop the following column: `TransactionId`, `BatchId`, `AccountId`, `CustomerId`, `SubscriptionId`. This may be potential target leakage if we're trying to predict that an accountId is suspect to a fraud we need to know if the latter has been involved in the past with a fraud transaction.


## Naive approach

Removing unique IDs features yields definitely better results for the RandomForest and XGBoost model as shown in the following table:

| Model       | Public Score | Private Score | Dataset |
| ----------- | ----------- |---- |---|
| **RandomForest**      | **0.712**       | **0.705**  |**without unique ID features**
| Tree      | 0.643       | 0.596  |without unique ID features
| XGBoost      | 0.702       | 0.689  |without unique ID features

There is definitely have room for improvement and this lead us to analyse feature that have greater impact on the fraud result of a transaction. To do so we can evaluate feature using MI score, short for Mutual Information (MI) score. It is a measure used to quantify the dependence between two variables. The MI score essentially captures the amount of information one can obtain about one variable by observing the other variable.

The MI score ranges from 0 to infinity. A score of 0 indicates that the two variables are completely independent, meaning there is no relationship between them. As the score increases, it indicates a stronger dependency between the variables. Higher MI scores suggest that the relationship between the variables can potentially provide more useful information for building predictive models.

In our context we will evaluate features dependency regarding the target value being the fraud result.

## Feature Analysis
As a first step we can investigate how the mean and standard deviation value evolve regarding the following features: `ProductId`, `Expense`, `ProviderId`, `ProductCategory`, `ChannelId`, `PricingStrategy`.

As some of those features are categorical, it is recommended to use One-hot encoding. It is a technique used to convert categorical variables into numerical format by creating binary columns (0s and 1s) for each category of the variable. When the categorical variable has no inherent order, such as colors or cities, one-hot encoding is appropriate. Since there is no meaningful way to rank or order these categories, creating separate binary columns ensures that the machine learning model does not assume any ordinal relationship between them.

One-hot encoding is only feasible when the categorical variable has a limited number of distinct categories. If the variable has too many categories, it can result in a large number of columns, which may lead to increased memory usage and longer training times. In our case categorical features are `ProviderId`, `ProductCategory`, `ChannelId` and `PricingStrategy` that contains respectively 6, 9, 4 and 4 unique values which are low enough to be one-hot encoded.

Furthermore, some machine learning algorithms, such as linear regression or logistic regression, require numerical inputs and may not handle categorical data directly. One-hot encoding helps transform categorical data into a format that these algorithms can work with.

After some data maninulation and programming we have the following MI score for our 44 features:

[![](https://hackmd.io/_uploads/S1CtYvB4h.png)
](https://)

## Results
We definitely see that from `ChannelId_1` and below features the MI scores is really low and is more likely to induce missleading to our model. We can see the difference between keeping all features, feature with a MI score greater than 0.001 and finally features with MI score greather than 0.0001 which yield the best result with precission rate above 78% as shown in the following table:


| Model       | Public Score | Private Score | Dataset |
| ----------- | ----------- |---- |---|
| RandomForest      | 0.640       | 0.609  |With all features
| TreeClasifier      | 0.643       | 0.596  |With all new added features
| XGBClassifier      | 0.704       | 0.655  |With all new added features
| RandomForest      | 0.654       | 0.631  |Features with MI Scores > 0.001
| TreeClasifier      | 0.654       | 0.631  |Features with MI Scores > 0.001
| XGBClassifier      | 0.688       | 0.704  | Features with MI Scores > 0.001
| **RandomForest**      | **0.786**       | **0.707**  |**Features with MI Scores > 0.0001**
| TreeClasifier      | 0.643      | 0.596 |Features with MI Scores > 0.0001
| XGBClassifier      | 0.704    | 0.655  | Features with MI Scores > 0.0001

# Unbalance Dataset

## K-Mean Undersampling

## SMOTE Oversampling

## Results

# Conclusion


Model used
Decision Tree Classifier
Random Forest Classifier
XGBoost Classifier
Logistic Regression (in the SecondModel.ipynb, wasn't good so stop)
