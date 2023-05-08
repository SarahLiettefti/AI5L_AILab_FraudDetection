# AI5L Artificial Intelligence Report

## Student
- LIETTEFTI LENS Sarah [18253]
- MITROVIC Nikola [18365]

# Introduction

In today's digital age, the rapid growth of online transactions has brought about an alarming increase in financial fraud. Detecting fraudulent transactions is crucial to maintain the security and trust of customers and financial institutions. The objective of this report is to provide an in-depth analysis and evaluation of machine learning models for the [Xente Fraud Detection Challenge](https://zindi.africa/competitions/xente-fraud-detection-challenge), a dataset containing transaction data from Xente, an e-commerce platform in Africa.

This is a case of binary classification, meaning that the model has to accurately classify input data into one of the two classes based on a set of features. The dataset contains transactions and the model has to predict if it is a fraud or not. 

This report will present a detailed analysis of the Xente transaction dataset, including data preprocessing, feature engineering techniques employed, and the rationale behind each decision. Furthermore, we will evaluate the performance of our chosen model, comparing it with various other models to showcase its strengths and limitations. Finally, we will emphasize the importance of scientific reasoning and its role in enhancing the overall effectiveness of our fraud detection model.

By the end of this report, we aim to provide a comprehensive understanding of the steps involved in developing a robust fraud detection model, as well as insights into the performance and suitability of different machine learning techniques for this task. Our findings will contribute to the ongoing efforts to improve the security of online transactions and mitigate the risks associated with financial fraud.


## Specificity of the dataset
The present study reveals an unbalanced training dataset where one class is substantially underrepresented compared to the other. Specifically, the minority class corresponds to fraudulent instances, which are only represented by 193 data samples, while the majority class encompasses 95469 non-fraudulent samples. While this is a common issue in real-life scenarios, it can lead to a model that is overly sensitive towards the majority class and produce biased and inaccurate predictions.

![](https://hackmd.io/_uploads/BJhusXBN3.png)


To tackle this challenge, we have identified two main approaches:

1. Using a model that accounts for the imbalanced data by assigning weights to each class. By assigning a positive weight to a class, it increases the cost for misclassifying instances of that class during training. This approach enables the model to pay more attention to the minority class, potentially improving its ability to correctly classify instances of it. Notably, Random Forest and XGBoost are examples of such models that will be implemented in this project. The optimal positive weight is calculated using the square root of the ratio of non-fraudulent to fraudulent instances (i.e., $\sqrt{N_{nonfraud} \over N_{fraud}}$), and it was found to be 22. We have tested several weights using the XGBoost model in the [SecondModels.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/SecondModels.ipynb), and show the best performances. 

2. Employing data sampling techniques to balance the number of instances in each class. Two main sampling methods are available: undersampling and oversampling.
    - **Undersampling** entails removing instances from the majority class. However, this approach may not perform well if the remaining data is insufficient for the model to learn from. In this study, undersampling resulted in only **386 instances**, which is inadequate. We tested both random and intelligent undersampling methods. Specifically, we utilized k-means to generate **193 clusters** from the majority class and selected one data point from each cluster. However, both methods yielded poor performances with a resulting score of approximately **0.007513148** obtained from the website.
    - **Oversampling**, in contrast, involves generating new data for the minority class to balance the dataset. This method showed better results compared to undersampling. We also tested random and intelligent oversampling using **SMOTE (Synthetic Minority Oversampling Technique)**. SMOTE was selected as it interpolates new instances between existing minority class instances, thereby avoiding overfitting.


All these methods were tested in the [TestEverything.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/TestEverything.ipynb).

The result were balanced but the best that were tested on the website were :
| Model       |Positive weight |Description|Public Score | Private Score |
| ----------- |-----------|-----------| ----------- |---- |
| XGBoost Classifier|22|Without the columns with very low MI|0.711864| 0.715447 
| Random Forest Classifier|None but used SMOTE to balance the data|36 decision three and the criterion = Entropy|0.709677|  0.686567
| Random Forest Classifier| 22|Basic data and 36 decision three and the criterion = Entropy| 0.703704       | 0.654867	
| Random Forest Classifier  |500| Basic data and 36 decision three and the criterion = Entropy| 0.678571       | 0.666667

The initial model, which utilized basic features engineering, obtained a score of 0.78 on the website. However, despite making efforts to improve the model, we were unable to replicate this score.

### How we evaluate
Using the appropriate metrics is crucial to accurately evaluate the performance of our models. Given the binary classification task and unbalanced dataset in our study, we employed five metrics, including the mean of the five metrics, to assess model performance. These metrics are fully explained in the notebook.k [EvaluationMetric.ipynb](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/EvaluationMetric.ipynb). 


We use those to have an idea of the performance of the models. We always split the data in training set and validation set before fitting it to the model to avoid target leakage.
1. Precision : $TP \over (TP+FP)$ It shows how sensitive is the model to the signal to be recognized. So how often we are correct when we classify a class positive. More close to 1 it is the better. A high-precision model, means that it doesn't always recognize a fraud, but when it predicts it,  it is certain of it. 
2. Recall : $TP \over (TP+FN)$ A model with a high recall will recognize as many positive classes as possible. We want a high recall if we want to be sure to detect all the fraud and don't care that sometimes it classifies non fraud as frauds. (Includes noises)
3. F1 score : combines precision and recall (2 complementary metrics) into one metric. $2 * {precision*recall} \over {precision+recall}$ It is probably the most used metric for evaluating binary classification models. If our F1 score increases, it means that our model has increased performance for accuracy, recall or both.
4. Log loss : This metric measures the difference between the probabilities of the model’s predictions and the probabilities of observed reality. The goal of this metric is to estimate the probability that an example has a positive class. More the log Loss is near 0 better it is.
5. Matthews Correlation Coefficient (MCC) : is designed to evaluate even models trained on unbalanced dataset. It ranges between $-1$ and $+1$. We want it to be near $+1$ since it indicates a correlation between actual observed values and predictions made by our model.


The mean of the computed metrics, except for the Log Loss metric, is used to provide a general idea of the model's performance since the Log Loss value is close to zero. The accuracy metric is not used in this study as it is not recommended for unbalanced data since a model that predicts only one class can have a high accuracy while failing to predict the minority class.

These metrics are employed to obtain a sense of model performance. Once we are satisfied with the model's performance, we use it to predict values from the test dataset (excluding the target variable) and evaluate it on the website. Notably, the scores obtained on the website are usually around **0.66666**, while our metrics indicate scores ranging between **0.8** and **0.98** for the same models. This difference could be attributed to overfitting, where the model may perform well on the training data but poorly on unseen data.


# Feature Engineering
## Preprocessing
We first need to set up a reference score with barely untouched data. Some processing might be still necessary in order to train a model. All the manipulation done on the training dataset is also done with the test dataset to have the same columns and data structure. 

As a first easy step, we can count the number of unique entries in each column of our training dataset, and then identifies columns with only one unique value. These columns can be dropped from both the training and testing datasets, as they do not provide any useful information for model training.

The next important step is to convert any object or string value into a numerical value. We first find and process columns containing `Id` in their names, as they contain string values. For example : the values of the column `BatchId` as this format : `BatchId_36123`. To make it trainable with our model, we remove the column name prefix and converts the values in these columns to integers (so it becomes `36123`).

We can also observe a timestamp for each transaction. As it is in string format, we need to perform some conversion to make this data usable for the training process. So we convert the `TransactionStartTime` column to a datetime format, enabling the extraction of various time-based features. After some quick observation, we see that transactions are spread across 4 months between 2018 and 2019. So keeping the month or the year is not useful, as we don't have any transaction for the other months or years. Moreover, it could teach the model that fraud only happens during those periods, and we want to avoid that. Still we can create new columns such as `TransactionDayOfWeek`, `TransactionDayOfMonth`, `TransactionHour`, `TransactionMinute` and `TransactionWeekofthemonth`. These features may help in capturing time-based patterns in the data. After the extraction, the original `TransactionStartTime` column must be dropped to not interfere with the new columns.

Following that, the `ProductCategory` column must be factorized as it contains string value for each category, which means the categorical data is encoded into numerical values. The factorized column is then converted to an integer data type for efficient processing.

Some models behave differently regarding discrete value (usually represented integer type) and continuous value (represented in float type). Therefore, we should convert `Value` column from integer to a float data type to avoid any problem later on for the training process. 

Finally, to avoid redundant information with `Value` column (absolute value of `Amount` column), we can create a new binary column called `Expense`, which indicates whether a transaction is an expense (Amount is negative) or not. The `Amount` column is then dropped, as its information is now captured by the `Expense` column.
We also tested if there were missing values in the dataset, and there were not.

## Reference Score Evaluation

We can now evaluate on Xente website how our base preprocessed dataset performs with different classifier models such as RandomForestClassifier, TreeClasifier and XGBoostClassifier. We will keep training on those three models in this report in order to measure and compare performance with different dataset manipulation, part of the feature engineering process. 

This yield the following result which we can base our future training to improve our model :

| Model       | Public Score | Private Score | Dataset |
| ----------- | ----------- |---- |---|
| RandomForest | 0.666       | 0.649  | Stock
| TreeClasifier | 0.678       |  0.666 |Stock
| XGBClassifier      | 0.666       | 0.682  |Stock

After conducting some investigation and using the feature description provided by Xente, we discover that some features have almost unique identifiers, including `TransactionId`, `BatchId`, `AccountId`, `CustomerId`, and `SubscriptionId`. However, including these columns in the model may result in target leakage, where the model is trained using information not yet available during prediction, such as a customer who previously committed fraud. While this can improve fraud detection for specific customers, it is not useful for new customers. Therefore, we drop these columns to avoid target leakage. We tested models with and without these columns to compare their performance. We can see in the following table the number of unique values for the mentionned features:

|Column|Unique Value|
| ----------- | ----------- |
|TransactionId |95662 |
|BatchId|94809|
|AccountId |3633|
|SubscriptionId|3627|
|CustomerId|3742 |


## Naive approach

Removing unique IDs features yields definitely better results for the RandomForest and XGBoost model as shown in the following table:

| Model       | Public Score | Private Score | Dataset |
| ----------- | ----------- |---- |---|
| **RandomForest**      | **0.712**       | **0.705**  |**without unique ID features**
| Tree      | 0.643       | 0.596  |without unique ID features
| XGBoost      | 0.702       | 0.689  |without unique ID features

There is definitely room for improvement and this lead us to analyze feature that have greater impact on the fraud result of a transaction. To do so, we evaluate the feature using MI score, short for Mutual Information (MI) score. It is a measure used to quantify the dependence between two variables. The MI score essentially captures the amount of information one can obtain about one variable by observing the other variable.

The MI score ranges from 0 to infinity. A score of 0 indicates that the two variables are completely independent, meaning there is no relationship between them. As the score increases, it indicates a stronger dependency between the variables. Higher MI scores suggest that the relationship between the variables can potentially provide more useful information for building predictive models.

In our context, we will evaluate features dependency regarding the target value being the fraud result, as shown in the [EvaluationFeatures.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/EvaluationFeatures.ipynb).
![](/imageFraud/NaiveApproch.png)

Later on, we will evaluate how our models perform when removing certain features. The goal is to identify noise in the data that may not be useful for fraud prediction. By removing noisy features, we can reduce the dimensionality of the data and prevent the useful features from being overshadowed, which may make it difficult for the model to learn from the data.


## Feature Analysis
In this step, we attempt to create new features and evaluate their impact on model performance. We also handle categorical features in this stage.

As a first step, we investigate how the mean and standard deviation values of the following features change: `ProductId`, `Expense`, `ProviderId`, `ProductCategory`, `ChannelId`, `PricingStrategy`.

As some of those features are categorical, it is recommended to use One-hot encoding. This technique is used to convert categorical variables into numerical format by creating binary columns (0s and 1s) for each category of the variable. When the categorical variable has no inherent order, such as colors or cities, one-hot encoding is appropriate. Since there is no meaningful way to rank or order these categories, creating separate binary columns ensures that the machine learning model does not assume any ordinal relationship between them.

One-hot encoding is only feasible when the categorical variable has a limited number of distinct categories. If the variable has too many categories, it can result in a large number of columns, which may lead to increased memory usage and longer training times. In our case categorical features are `ProviderId`, `ProductCategory`, `ChannelId` and `PricingStrategy` that contains respectively 6, 9, 4 and 4 unique values which are low enough to be one-hot encoded.

Furthermore, some machine learning algorithms, such as linear regression or logistic regression, require numerical inputs and may not handle categorical data directly. One-hot encoding helps transform categorical data into a format that these algorithms can work with.

After some data maninulation and programming we have the following MI score for our 44 features:

![](/imageFraud/FeatureAnalysisMI.png)

## Results
We can observe on the previous figure that for `ChannelId_1` and below features, the MI score is low and more likely to lead to misleading results in our model. We compare the performance of models trained with all features, features with an MI score greater than 0.001, and features with an MI score greater than 0.0001. The latter approach yields the best results, with a precision rate above 78%, as shown in the following table:


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

Here is Xente score screenshot:

![](/imageFraud/Results078.png)
The result mentioned above was obtained using the unbalanced dataset. (If you don't beleive us test it by yourself ;p this is the link to the file.)[https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/main/RandomForestBestResult.csv]

# Models
We used four model for this project : Decision Tree Classifier, Random Forest Classifier, XGBoost Classifier, Logistic Regression. The optimisation of these model was done in the [SecondModels.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/SecondModels.ipynb) where we tested several parameters to find the best ones. We then evaluated and compared several models at different stages of the feature engineering process in the [TestEverything.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/TestEverything.ipynb).


## Decision Tree Classifier 
*This section on decision tree does not deal with unbalanced data.*
The parameter we played with for the decision tree classifier is the maximum leaf nodes. Chosing the correct one help to avoid overfitting a model. It was done by training models with different `max_leaf_nodes` and evaluated with our metrics. We the took the best performing ones and tested with the website metrics.

![](/imageFraud/DecisionTreeClassifierGraph.png)

We can see a level between 22 and 28. We also wanted to see what what was happening around 4 and 7. This were the results : 

![](/imageFraud/DecisionTreeClassifierResult.png)

We choose to continue testing the decision tree model with maximum 6 leaf. We were surprise by the website result not being bad, knowing that we didn't deal with the unbalanced data yet and only did minimal feature engineering.

1. This is the score after doing more featuring engineering (with unbalanced data): 
2. We then tested it with deleting the feature with a low MI score and saw how it got much worst. 
3. We then tried to balance the dataset undersampleing with K-Means. We created 193 cluster of the majority class and choose a value of each cluster. This gave us a perfect exemple of a overfitting.  
4. The SMOTE was used on it but it performed badly.

|n|Model|Description|PublicScore|PrivateScore|Precision|Recall|F1-score|LogLoss|Mcc|MeanOurMetrics|
|--|---|---|---|---|---|---|---|---|---|---|
|1|DecisionTreeClassifier|Delete only very low + Niko Datas max_leaf_nodes=6|0.642857|0.596491|0.916667|0.785714|0.846154|0.018085|0.848426|0.849240|
|2|DecisionTreeClassifier|Delete low MImax_leaf_nodes : 6|NaN|NaN|0.500000|0.050|0.090909|0.060284|0.157730	|0.199660|
|3|DecisionTreeClassifier|Kmeans UnderSampling max_leaf_nodes : 6 |0.006575|0.006863|0.980392|1.00|0.990099|0.371584|0.979557|0.987512|
|4|DecisionTreeClassifier|SMOTE UpperSampling max_leaf_nodes : 6 |/|/|0.113372|0.975|0.203125|0.461171|0.330225|0.405431|

At this point, we also saw that and running the same code and training the same model could sometime give us very different data even though we allways use the same random_state. 

## Random Forest Classifier
We tried to optimze its criterion and number of tree parameters. We later tried to optimize the positive weight we could use to deal with the unbalanced data.

![](/imageFraud/RandomForestC1.png)
![](/imageFraud/RandomForestC2.png)
![](/imageFraud/RandomForestC3.png)
![](/imageFraud/RandomForestC4.png)

After evaluation, we concluded that it was best to use the Entropy or log loss criterion (and avoid the gini) and to limit the Forest to 36 trees. 

1. In the table below, we see that not dealing with the unbalanced data give a better PublicScore than upsampling it randomly. 
2. Upsampling is better than undersampling for the resaon discussed earlier
3. Unersampling results in overfitting
4. Have better result when giving a positive weight of 22 to balance the data.


|n|Model|Description|PublicScore|PrivateScore|Precision|Recall|F1-score|LogLoss|Mcc|MeanOurMetrics|
|--|---|---|---|---|---|---|---|---|---|---|
|1|Random Forest|Not dealing with unbalanced data|0.690909|0.649123|0.818182|0.900|0.857143|0.018085|0.857869|0.858298|
|2|Random Forest|Random UpSampling|0.678571|0.637168|0.765957|0.900|0.827586|0.022606|0.829975		|0.830880|
|3|Random Forest|Random UnderSampling|0.549020|0.480769|1|1|1|2.220446e-16|1|1|
|4|Random Forest|Weigth of 22 |**0.703704**|0.654867|0.804348|0.925|0.860465|0.018085|0.862324|0.863034|

Other evaluations can be find in the [TestEverything.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/TestEverything.ipynb).

## XGBoost Classifier
It is a model that can deal with unbalanced data with a positive weight. The optimization of it has been done in the [SecondModels.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/SecondModels.ipynb). The optimized one is 22 given by the following formula: $\sqrt{N_{nonfraud} \over N_{fraud}}$

Optimization of the positive weight : 

![](/imageFraud/XGBoostClasifier.png)

Here are more results : 
|n|Model|Description|PublicScore|PrivateScore|Precision|Recall|F1-score|LogLoss|Mcc|MeanOurMetrics|
|--|---|---|---|---|---|---|---|---|---|---|
|1|XGBClassifier w22|with minimul feature engineering|0.677419|0.676923|0.822222|0.925|0.870588|0.016578|0.871874|0.872421|
|2|XGBClassifier w22|More feature engineering|0.666667|0.682540|0.880952|0.880952|0.880952|0.015071|0.880743|0.666667|0.880900|
|3|XGBClassifier w22|Cleaning low MI|**0.711864**|**0.715447**|0.465116	|0.500	|0.481928	|0.064805	|0.481344|	0.482097|

The third model show an example when the performance on the training set is not great but it performs well on the website score. This is also an exemple of model that we will run several time (with the exact same code) and have significantly different performance metrics each time. 

## Logistic regression
It was only tested with the random upsampling and undersampling and perform so poorly that we didn't went further.
![](/imageFraud/BJBBHI84h.png)
![](/imageFraud/LogisticRegression.png)



## K-Mean Undersampling
We already reviewed it in the models section but here is a summary : 

![](/imageFraud/Kmeans.png)

Undersampling is not a solution for our dataset because there is not enough data for the models to learn from it. It is a good example of overfitting, when the evaluation on the validation data is much better than on the test data. 

## SMOTE Oversampling
![](/imageFraud/SMOTE.png)

It gaves us some good results for the Random Forest Classifier but not for the others models. 


# Conclusion

## Lessons Learned
In this project, we found that scientific reasoning played a crucial role in enhancing the effectiveness of our fraud detection model. Surprisingly, we discovered that deleting noisy features sometimes yielded better results than dealing with unbalanced data. Additionally, we observed that incorporating a positive weight of 22 and using SMOTE were effective techniques to improve model performance. It is essential to optimize model parameters while also knowing when to stop to avoid overfitting. In some cases, **less can be more**, and removing certain features can actually enhance model performance, hence the importance of selecting the data we use.These findings provide valuable insights into the performance and suitability of various machine learning techniques for fraud detection. 

While solutions for handling unbalanced data, such as undersampling or upsampling, can improve model performance to some extent, they cannot perform miracles when dealing with extremely limited transaction data. In cases where the dataset is highly imbalanced, such as our dataset with only 193 frauds out of +95000 transactions. In real-life applications in finance, fraud is likely to be unbalanced compared to regular transactions, but large banks and companies typically have access to much larger datasets. Even if the fraud ratio is very low, such as 2%, having millions of transactions can still provide enough fraud data to train on, and this can significantly improve model performance. Therefore, having access to large and diverse datasets is critical for developing accurate fraud detection models in finance and other industries.


## Further Investigation
There is certainly room for improvement as there are still many features that we can engineer to improve model performance. For instance, we can investigate how the frequency of transactions for specific features, such as `ProductCategory` or `ProviderId`, may influence the target result.

Another idea could be to compute the cumulative sum per day per product category or an other categorical feature. It can capture the trend of transactions for each feature over time. This can potentially provide valuable information for fraud detection, as it may be the case that fraudulent transactions have different temporal patterns compared to legitimate transactions.

Additionally, we did not apply any rescaling during the training process since we had goo results, but it is something we should absolutely consider in future iterations. Rescaling using min max alogritm for example can normalize the range of values in different features, which  is a crucial step as it can prevent certain features from dominating the learning process, improve model convergence, and make features more comparable, even if they are measured on different scales or units. Additionally, some optimization algorithms used in AI, such as gradient descent, can converge more quickly when input features are normalized. Therefore, rescaling features is highly recommended in AI and should be considered in any machine learning project.
