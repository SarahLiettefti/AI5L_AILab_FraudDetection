# AI5L Artificial Intelligence Report

# Introduction

In today's digital age, the rapid growth of online transactions has brought about an alarming increase in financial fraud. Detecting fraudulent transactions is crucial to maintain the security and trust of customers and financial institutions. The objective of this report is to provide an in-depth analysis and evaluation of machine learning models for the [Xente Fraud Detection Challenge](https://zindi.africa/competitions/xente-fraud-detection-challenge), a dataset containing transaction data from Xente, an e-commerce platform in Africa.

This is a case of binary classification, meaning that the model has to accurately classify input data into one of the two classes based on a set of features. The dataset contains transactions and the model has to predict if it is a fraud or not. 

The challenge involves developing an effective fraud detection model that can identify potentially fraudulent transactions by analyzing and learning from historical transaction data. To accomplish this task, we will explore various aspects of the dataset, perform data preprocessing and feature engineering to extract relevant information, and compare the performance of different machine learning models to identify the most suitable approach.

This report will present a detailed analysis of the Xente transaction dataset, including data preprocessing, feature engineering techniques employed, and the rationale behind each decision. Furthermore, we will evaluate the performance of our chosen model, comparing it with various other models to showcase its strengths and limitations. Finally, we will emphasize the importance of scientific reasoning and its role in enhancing the overall effectiveness of our fraud detection model.

By the end of this report, we aim to provide a comprehensive understanding of the steps involved in developing a robust fraud detection model, as well as insights into the performance and suitability of different machine learning techniques for this task. Our findings will contribute to the ongoing efforts to improve the security of online transactions and mitigate the risks associated with financial fraud.


## Specificity of the dataset
The training dataset is unbalanced, meaning that there is a minority class that is significantly  less represented in the dataset than the other class. In this case, the minority class is the frauds with only 193 data compared to 95469 non-frauds. Although this is expected in real-life scenarios, it can lead to an overly sensitive model towards the majority class and produce biased and inaccurate predictions.
![](https://hackmd.io/_uploads/BJhusXBN3.png)
To address this issue, we have two main options:
1. Using a **model** that take into account the imbalanced data by assigning **weights** to each class. By setting a positive weight for a class, it increases the penalty for misclassifying instances of that class during training. Making the model pay more attention to the minority class and potentially improving its ability to correctly classify instances of it. Random Forest and XGBoost are examples of such models that will be used in this project. We found that the optimal positive weight is 22. It was calculates by the $\sqrt{N_{nonfraud} \over N_{fraud}}$ . Several weights have been tested with XGBoost model in the [SecondModels.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/SecondModels.ipynb), and this one had the best performances. 
2. **Sampling** the dataset to balance the number of instances in each class. There are two types of sampling methods: **undersampling** and **oversampling**. 
    - **Undersampling** involves removing instances from the majority class, but this may not perform well when the remaining data is not sufficient for the model to learn from. In this case, it only left us with 386 instances, which is not enough. The result from the website were around 0.007513148. We can sample randomly or with some intelligence behind it. For the not random sampling, we used k-means to create 193 clusters of the majority class and select 1 data from each cluster. Both case were tested and had terrible performances.
    - **Oversampling**, on the other hand, involves creating new data for the minority class, to balance the dataset. This had a better result (see later). We also tested to sample randomly and with some intelligence behind it with SMOTE (Synthetic Minority Oversampling Technique). This method of SMOTE was chosen because it avoids overfitting by interpolating new instances between existing minority class instances. 

All these methods were tested in the [TestEverything.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/TestEverything.ipynb).

The result were balanced but the best that were tested on the website were :
| Model       |Positive weight |Description|Public Score | Private Score |
| ----------- |-----------|-----------| ----------- |---- |
| XGBoost Classifier|22|Without the columns with very low MI|0.711864| 0.715447 
| Random Forest Classifier|None but used SMOTE to balance the data|36 decision three and the criterion = Entropy|0.709677|  0.686567
| Random Forest Classifier| 22|Basic data and 36 decision three and the criterion = Entropy| 0.703704       | 0.654867	
| Random Forest Classifier  |500| Basic data and 36 decision three and the criterion = Entropy| 0.678571       | 0.666667

We also got a 0.78 score (the website one) with very basic model and featuring engineering. But we couldn't reach this score again, even though we were doing everything to improve the model.

### How we evaluate
It is important to use the right metric to evaluate our models to have the right idea on its performance. In our case, we are doing a binary classification with unbalanced dataset. For that we are using five metrics (plus de mean of those). They are calculated and explain in the notebook [EvaluationMetric.ipynb](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/EvaluationMetric.ipynb). 


We use those to have an idea of the performance of the models. We always split the data in training set and validation set before fitting it to the model to avoid target leakage.
1. Precision : $TP \over (TP+FP)$ It shows how sensitive is the model to the signal to be recognized. So how often we are correct when we classify a class positive. More close to 1 it is the better. A high-precision model, means that it doesn't always recognize a fraud, but when it predicts it,  it is certain of it. 
2. Recall : $TP \over (TP+FN)$ A model with a high recall will recognize as many positive classes as possible. We want a high recall if we want to be sure to detect all the fraud and don't care that sometimes it classifies non fraud as frauds. (Includes noises)
3. F1 score : combines precision and recall (2 complementary metrics) into one metric. $2 * {precision*recall} \over {precision+recall}$ It is probably the most used metric for evaluating binary classification models. If our F1 score increases, it means that our model has increased performance for accuracy, recall or both.
4. Log loss : This metric measures the difference between the probabilities of the model’s predictions and the probabilities of observed reality. The goal of this metric is to estimate the probability that an example has a positive class. More the log Loss is near 0 better it is.
5. Matthews Correlation Coefficient (MCC) : is designed to evaluate even models trained on unbalanced dataset. It ranges between $-1$ and $+1$. We want it to be near $+1$ since it indicates a correlation between actual observed values and predictions made by our model.

The mean of those values (except of the Log loss because it is close to 0) is used to just have a general idea of the performance.

We didn't use the accuracy metric, even if it is a common and simple metric because it is not recommended for our case of unbalanced data. If the model only predicts 0, it can have an accuracy of 99,7% while having failed to predict all the frauds. 

Those metrics are used to have an idea of the performance. Once we are happy with the performance of a model, we use it to predict value from the test dataset (without the target) and evaluate it on the website. Usually the score on the website are around 0.66666 whereas for the same model with our metrics are between 0.8 and 0.98. This difference could be explained with the phenomena of overfitting. 


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

After some investigation and using the feature description provided by Xente, we see that some features have (almost) unique identifier. Those columns are : `TransactionId`, `BatchId`, `AccountId`, `CustomerId`, `SubscriptionId`. The problem with it is that the model could learn, for example, that a certain client, identified with his CustomerId, often do frauds. This is a good thing to predict fraud for this specific customer, but it doesn't help for the transaction of new customer. This could generate target leakage, training with an information we don't have yet (if the client fraud in the past). To avoid it, those columns could be dropped. We tested some models with those columns and without.

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

In our context, we will evaluate features dependency regarding the target value being the fraud result. You can see it in the [EvaluationFeatures.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/EvaluationFeatures.ipynb).
![](https://hackmd.io/_uploads/S1ZafrIV2.png)

We will later evaluate how our models perform when removing some features. The goal is to identified the noices. Those kind of data could be unuseful for the prediction, but by being present in the training of the model, it could take too much space and the useful features could be drown into it. Making it difficult for the model to learn from it. 


### Feature Analysis
This step is trying to create new features and evaluate if it improve the model performances. We are also dealing with the categorical features at this stage.

As a first step, we can investigate how the mean and standard deviation value evolve regarding the following features: `ProductId`, `Expense`, `ProviderId`, `ProductCategory`, `ChannelId`, `PricingStrategy`.

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

Here is Xente score screenshot:

![](https://hackmd.io/_uploads/B1zYsOSN3.png)
Those result were by using the unbalanced data. 

# Models
We used four model for thes projest : Decision Tree Classifier, Random Forest Classifier, XGBoost Classifier, Logistic Regression. The opitmisation of those are done in the [SecondModels.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/SecondModels.ipynb) were we tested several paramters to find the best. We then evaluated and compared several models with several stage of the feature engineering in the [TestEverything.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/TestEverything.ipynb).


## Decision Tree Classifier
(this is with unbalanced data)
The parameter we played with for the decision tree classifier is the maximum leaf nodes. Choising the correct one help to avoid overfitting a model. It was done by training models with different `max_leaf_nodes` and evaluated with our metrics. We the took the best performing ones and tested with the website metrics.

![](https://hackmd.io/_uploads/BydudHLV2.png)
We can see a level between 22 and 28. We also wanted to see what what was happening around 4 and 7. This were the results : 
![](https://hackmd.io/_uploads/SkaZYH8V2.png)

We choose to continue testing the decision tree model with maximum 6 leaf. We were surprise by the website result not being bad, knowing that we didn't dealt with the unbalanced data yet and only did minimal feature engineering. 

1. This is the score after doing more featuring engineering (but stillunbalanced data): 
2. We then tested it with deleting thefeature with a low MI and so how it got much worst. 
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
![](https://hackmd.io/_uploads/BJ8nRSUNn.png)
![](https://hackmd.io/_uploads/BJpn0SU42.png)
![](https://hackmd.io/_uploads/By1p0HUV2.png)
![](https://hackmd.io/_uploads/rk_CArU4h.png)
After evaluation, we conclude that it was best to use the Entropy or log loss criterion (and avoid the gini) and to limit the Forest to 36 trees. 

1. In the table below, we see that not dealing with the unbalanced data give a better PublicScore than upsampling it randomly. 
2. Upsampling is better thant under sampling for the resaon discuss earlier
3. Unersampling is a good case of overfitting
4. Have better result when giving a positive weight of 22 to balance the data.


|n|Model|Description|PublicScore|PrivateScore|Precision|Recall|F1-score|LogLoss|Mcc|MeanOurMetrics|
|--|---|---|---|---|---|---|---|---|---|---|
|1|Random Forest|Not dealing with unbalanced data|0.690909|0.649123|0.818182|0.900|0.857143|0.018085|0.857869|0.858298|
|2|Random Forest|Random UpSampling|0.678571|0.637168|0.765957|0.900|0.827586|0.022606|0.829975		|0.830880|
|3|Random Forest|Random UnderSampling|0.549020|0.480769|1|1|1|2.220446e-16|1|1|
|4|Random Forest|Weigth of 22 |**0.703704**|0.654867|0.804348|0.925|0.860465|0.018085|0.862324|0.863034|

Other evaluations can be find in the [TestEverything.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/TestEverything.ipynb).

## XGBoost Classifier
It is a model that can deal with unbalanced data with a positive waight. The optimization of it has been done in the [SecondModels.ipynb script](https://github.com/SarahLiettefti/AI5L_AILab_FraudDetection/blob/clean/SecondModels.ipynb). The optimized one id 22 : $\sqrt{N_{nonfraud} \over N_{fraud}}$

Optimization of the positive weight : 
![](https://hackmd.io/_uploads/SJMh-8U42.png)

Here are more result : 
|n|Model|Description|PublicScore|PrivateScore|Precision|Recall|F1-score|LogLoss|Mcc|MeanOurMetrics|
|--|---|---|---|---|---|---|---|---|---|---|
|1|XGBClassifier w22|with minimul feature engineering|0.677419|0.676923|0.822222|0.925|0.870588|0.016578|0.871874|0.872421|
|2|XGBClassifier w22|More feature engineering|0.666667|0.682540|0.880952|0.880952|0.880952|0.015071|0.880743|0.666667|0.880900|
|3|XGBClassifier w22|Cleaning low MI|**0.711864**|**0.715447**|0.465116	|0.500	|0.481928	|0.064805	|0.481344|	0.482097|

The third model show an example when the performance on the training set is not great but it perform well on the website score. This is also an exemple of model that we will run several time (with the exact same code) and have significantly diffrent performance metrics each time. 

## Logistic regression
It was only tested with the random upsampling and undersampling and perform so poorly that we didn't went further.
![](https://hackmd.io/_uploads/BJBBHI84h.png)
![](https://hackmd.io/_uploads/SJ-Lr8IE3.png)



## K-Mean Undersampling
We already reviewed it the models section but here is a summary : 
![](https://hackmd.io/_uploads/r15a3rUN2.png)
Undersampling is not a solution for our dataset because there is not enough data for the models to learn from it. It is a good example of overfitting, when the evaluation on the validation data is much better than on the test data. 

## SMOTE Oversampling
![](https://hackmd.io/_uploads/ByMqBIL42.png)

It gaves us some good results for the Random Forest Classifier but not for the others models. 


# Conclusion

Finally, we will emphasize the importance of scientific reasoning and its role in enhancing the overall effectiveness of our fraud detection model. 

We saw that sometimes, it is better to delete the noices features than even dealing with the unbalanced data (that was surprising). 
Apart for that, we saw that adding a positive weight of 22 was very effective to improve the performance of our models as well as using the SMOTE. 
It is important to optize the parameters of the models but we also need to learn when to stop to not go too far. Sometime doing less is more.

Insights into the performance and suitability of different machine learning techniques for this task.


