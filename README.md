# **Twitter NLP**

## **Authors**:
Bobby Daly, DS; Ben Gilbert, DS

![image](https://github.com/Ben-Gilbert333/Twitter_NLP/assets/126971652/2744d4f2-ed49-426e-802e-4dcc8f05f335)                           ![image](https://github.com/Ben-Gilbert333/Twitter_NLP/assets/126971652/52bdc3d7-82fe-4880-85b3-3e7a797f5415)



## **Overview**
For this project we analyzed over 8,000 tweets from the 2011 SXSW festival. The dataframe used consisted of the text in the tweet, the sentiment of the tweet (rated by a human), and the product/brand the tweet was directed towards. Before modeling we performed a preprocessing approach that included:

* standardizing
* removing stopwords
* tokenizing
* stemming
  
Then we began to build models that aimed to predict the sentiment associated with each tweet. Our final model is 68% accurate. This is 8% more accurate than our dummy model. Throughout this process we reached the conclusion that designing products more similar to Apple's will receive better reactions from the public. Next steps include creating a binary model, using our model to analyze tweets after the 2024 SXSW event, and improving data collection.


## **Business Problem**
We are Orange, a tech company, looking to improve our next product release at SXSW 2024. This festival is a great opportunity for major tech companies like Google, Apple and now Orange, to reveal new products and obtain a good public image. By analyzing tweets from a previous SXSW event and the sentiment associated with each tweet we attempt to build a model that can predict the sentiment given text from a tweet. This will help us in the future understand how our new products are being recepted by the public.

For this project we chose accuracy as the metric to focus on. We decided that positive, negative, and neutral tweets all add value to determining how to handle product design in the future based on sentiment towards previous products. Given this train of thought we didn't find it necessary to optimize any class over the other and thought accuracy score would be best to use in order to evaluate our models.

## **Data Understanding**
![image](https://github.com/Ben-Gilbert333/Twitter_NLP/assets/126971652/1f99767d-f808-4f18-aed4-5600a1cfc0d8) <br>
The data comes from CrowdFlower via [data.world](https://data.world/crowdflower/brands-and-product-emotions). It consists of 9,000 tweets about Apple and Google products from a South by Southwest (SXSW) event. Human raters rated the sentiment of the tweets as positive, negative, neutral, or indistinguishable. There are 3 columns including the tweet, the product the tweet is about, and the sentiment of the tweet. The variable we used as the target is sentiment. Our goal is to find key words in tweets that can be used to identify the sentiment of each tweet.

## **Data Preparation**
The steps we took to prepare our data before NLP processing included:
* dropping a column
* dropping a null
* renaming a column
* dropping tweets with 'I can't tell' as sentiment

The NLP steps performed included:
* standardizing
* tokenizing
* removing stopwords
* stemming

Created three bar charts that display the top 10 most common tokens in each sentiment group (positive, negative, and neutral). These charts are how we decided which tokens needed to be added to our stopwords list. If a token was prevelant in all sentiment groups we added it to our stopwords list because these tokens do not add any value in determining sentiment. This process was repeated until we felt the top tokens were diverse and meaningful.

Paying attention to the class imbalance is important for this process. For example, in this final iteration of the charts 'social' appears as a top token in the neutral and negative groups. The scales between these groups is substantial: 'social' appears about 30 times in the negative group and nearly 500 times in the neutral group. Due to this difference we decided not to add 'social' to our stopwords list.

<img width="874" alt="Screenshot 2023-09-14 at 3 42 11 PM" src="https://github.com/Ben-Gilbert333/Twitter_NLP/assets/126971652/52093f44-c227-48fc-8459-0d90f48924d2">



## **Modeling**
In order to analyze the performance of models we built, we focused on accuracy. We decided this was best because we felt negative and positive sentiment towards a brand/product are equally important. The negatives inform what needs to be changed and what could be different. Positives inform what should stay the same and the strengths of a company/product. If we chose recall or precision we would be focusing more on one than the other.

For this problem we needed to use classification models. There are many options to use and we weren't sure which would be most appropriate so we created a wide variety. First, we started with a decision tree purposefully trying to overfit the model to confirm that we had the right data in order to build a successful predictive model.

Every iteration of our models was highly focused on reducing overfitting while maintaining or increasing the accuracy score.

### **Evaluation**
#### **Baseline Understanding**
The most dominant sentiment in the data is neutral, therefore we set our baseline accuracy as a model predicting neutral sentiment every time. This would result in an accuracy score of approximately 60%.

#### **First Model (Decision Tree)**
We created a decision tree model with all default parameters. Our goal here was to purposefully create an overfit model in order to confirm it achieves a high accuracy score. Accomplishing this tells us we are using good enough data in order to predict our target variable. This worked successfully, this model received an accuracy score of 95.17% on the training data. But, the cross_val_score is much lower (64%) showing this model is very overfit. This is still performing a little better than the baseline model.

#### **Second Model (Random Forest)**
We then created a Random Forest model with default parameters and added max_features=2000 to the vectorizer in order to help reduce overfitting. Again, this random forest model earned a high accuracy score (95%) on the training data but it is most likely due to overfitting. The cross validation score (67%) is a bit higher than it was for the decision tree model. This random forest model is still very overfit to the training data.

#### **Random Forest with GridSearch**
In order to combat the overfit results we used a GridSearch to help tune the hyperparameters. The GridSearch discovered max_depth=75, max_features=1000, and n_estimators=18 had the best cross validation score. This is almost the same score as before but should be much less overfit considering the hyperparameters. After two rounds of paramter tweaking the GridSearch converged on a cross validation score of 68%, 1% higher than our Random Forest without a GridSearch!

#### **Third Model (Logistic Regression)**
We then created a logistic regression model with an awareness that logistic regression does not handle class imbalances as well as decision trees/random forests. We used max_features=900 in the vectorizer because it worked the best from the GridSearch with the random forest model. This logistic regression has an 75% accuracy score which is pretty high but it a cross validation score of 67% shows that it is overfit, but not as bad as the base random tree or base random forest models. We then created confusion matrix in order to investigate how the model is handling the class imbalance. It shows that this model is struggling predicting the negative and positive sentiments accurately but is doing pretty well with the neutral sentiment becuase most of the data is neutral. <br>
<img width="362" alt="Screenshot 2023-09-14 at 4 18 07 PM" src="https://github.com/Ben-Gilbert333/Twitter_NLP/assets/126971652/17dfaf8b-17e3-4fe9-bc06-b861bd995f87"> <br>

#### **Logistic Regression with GridSearch**
In order to combat the poor performance due to overfitting and the class imbalance, we used a GridSearch to help tune the hyperparameters. Using a class weight 'balanced' should help with the class imbalance problem, lower C will enforce stricter regularization helping with overfitting. The GridSearch discovered C=0.1, max_iter=5000, balanced weights, and solver='lbgfs' had the best cross validation score of 58%. This is a lot worse than our logistic regression with all default settings. We expected this becuase we reduced overfitting.

#### **Fourth Model (Multinomial Naive Bayes)**
Created a Multinomial Naive Bayes (MNB) model. Kept the max_features=900 in the vectorizer to reduce overfitting again. This was a lower accuracy score (71%) on the training data than on previous models that sparked hope it was not overfit. A cross validation score of 67% showed us that this model was a little less overfit than the previous models, but still was overfit on the training data.

#### **Logistic Regression with GridSearch**
We tried to improve the MNB model by adjusting hyperparameters using a GridSearch. This did not improve much from the accuracy score using the defualt parameters, but it did improve cross validation by about 0.5%.

#### **Final Model (Stacking Random Forest, Logistic Regression, and MNB Models)**
Finally, we made an improved model by stacking our best iterations of the previous models. Doing this resulted in an accuracy score of 81% on the training data and a cross validation score of 68.35%. We spent a lot of time changing the hyperparameters on this StackedClassifier model. Nothing improved the cross validation score. We tried changing the final_estimator, max_features in the vectorizer, and much more.

We decided to use this as our final model and use it on the testing data. It is an overfit model but so were all of the others and it received the highest cross validation score. However, it was a bit disappointing considering it is only about 8% more accurate than our dummy model.

## **Conclusions**
When deployed on the testing data it received an accuracy score of 67% this is comparable to the cross validation score and was expected. There were two huge challenges when creating this model:

1. Combat overfitting
2. Dealing with the class imbalance
   
There were a tiny amount of negative tweets compared to neutral tweets. There were more positive tweets than negative but, not nearly as many positive tweets as neutral tweets. Our model struggled predicting negative and positive tweets as shown by this confusion matrix. Majority of the time the model predicts neutral because of this class imbalance. <br>
<img width="388" alt="Screenshot 2023-09-14 at 4 29 40 PM" src="https://github.com/Ben-Gilbert333/Twitter_NLP/assets/126971652/81c7ba65-21b0-4ef0-809a-035bb611b598"> <br>

## **Next Steps**
After our research, we recommend creating another model attempting to solve the same problem but, create a binary classification problem using positive and negative. Neutral tweets do not add much value in the big picture and created a distinct class imbalance making it difficult to create a successful model.

In the meantime, we could still use this model at SXSW 2024 in order to obtain knowlegde on public sentiment of our products that we launch at the event.

Improving the data collection process would also be useful. The data we had access to had many nulls in the product column. We were able to distinguish between Apple and Google with the data at hand. With improved data we could get more granular and discover which specific products the sentiment is directed towards rather than just the company.


## **Thank You**
Thank you for taking the time to review our research.
We hope this information helps and we look forward to working with you more on the next steps.

Sincerely, <br>
Bobby Daly, Ben Gilbert <br>

## Further Details
Further details are available in the full analysis presented in the [Jupyter Notebook](https://github.com/Ben-Gilbert333/Twitter_NLP/blob/main/Twitter_NLP_Analysis.ipynb). 

## Repository Structure
```
├── data
├── notebooks
├── README.md
├── LICENSE
├── .gitignore
├── Twitter NLP Analysis Presentation.pdf
└── Twitter NLP Analysis.ipynb
```


