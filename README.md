# Recipes and Ratings classifier model

By Pin-Hsuan Lai (pilai@ucsd.edu), Pin-Yeh Lai (p2lai@ucsd.edu)

## Framing the Problem

Our dataframe is basically following the steps of data cleaning in project 3, so most of the code is the same, but there are still some new cleaning added.

In project 3, our data cleaning process: We first download the two required files `RAW_recipes.csv` and `RAW_interactions` (These dataset contains recipes and ratings from *food.com*. It was originally scraped and used by the authors of *Generating Personalized Recipes from Historical User Preferences* recommender systems paper). And then merge them together with recipes as left. We then replace all the zeros in the column `rating` with np.nan, then get the average rating of each id and add it to the dataframe as a new column, and then started to clean the whole dataframe. We use eval to transform the columns whose content can be made into a list into columns of list, and then break the column `nutrition` into the corresponding columns. Finally, we convert the time of `submitted` and `date` into timestamp and finish the cleaning of the dataframe.

We also have some new steps, we create two new columns. The first column is `Before 2013`, we use apply and lambda create this column which stated is the review is made before 2013 or not. The other column is `submitted_year`, which is the year of `submitted` timestamp column, generate also by apply and lambda. Finally, the last step, We remove rows that contain missing values in the column `ave_rating`. We consider the percentage of missing values in `ave_rating` to be only 1% of the entire dataset, and removing these data will not have a substantial impact on the prediction results.

Our prediction problem involves using recipe information to predict the year of each review, specifically whether the review occurred before or after 2013.

This prediction problem is binary classification, with only two possible outcomes: True and False, representing before or after 2013, respectively. During Project 3, we discovered that there are performance differences between reviews before and after 2013. In our dataset, we have data of recipe reviews, ratings, calories, and other features from the years 2008 to 2018. Therefore, we are able to use these data features from the years 2008-2018 to train our model.

The metric we have chosen to evaluate our model is accuracy. This decision is based on the fact that false positives and false negatives in our prediction have no significant differences in consequences. There is no greater consequence associated with one over the other. Therefore, precision and recall are not as crucial, and the most important factor is the overall correctness of the predictions. Hence, accuracy is the best choice for evaluation.

## Baseline Model

In our data, True represents reviews from 2013 and earlier, while False represents reviews from after 2013.

**The distribution of reviews that done before 2013 and after 2013:**

| index   |   Before 2013 |
|:--------|--------------:|
| True    |      0.820127 |
| False   |      0.179873 |

In our baseline model, we decided to use columns `calories`, `sugar` and `protein` as our independent variables to train our model, since we can observe in below three graph that the mean of all three columns across two time period are different.(These difference may or may not are significant).

**Graph 1. The mean of `calories` before and after 2013:**

<iframe src="assets/Mean Calories Before and After 2013.html" width=800 height=600 frameBorder=0></iframe>

We can see obviously there is a gap between mean of calories before and after 2013.

**Graph 2. The mean of `sugar` before and after 2013:**

<iframe src="assets/Mean Sugar Before and After 2013.html" width=800 height=600 frameBorder=0></iframe>

We can see obviously there is a gap between mean of sugar before and after 2013.

**Graph 3. The mean of `protein` before and after 2013:**

<iframe src="assets/Mean Protein Before and After 2013.html" width=800 height=600 frameBorder=0></iframe>

We can see obviously there is a gap between mean of protein before and after 2013.

- Column `calories` is a numerical column, we decide to input it to the model without change.

- Column `protein` is a numerical column, we decide to input it to the model without change.

- Column `protein` is a numerical column, we decide to input it to the model without change.

All three columns we select in baseline model are numerical, which are quantitative.

In the baseline model, we did not make any modifications to these columns and directly used them for training the model.

Our baseline model with DecisionTreeClassifier had the best hyper-parameters: {'Tree__max_depth': 20,'Tree__min_samples_leaf': 17, 'Tree__min_samples_split': 16}. 
```
Pipeline(steps=[('Tree',
                 DecisionTreeClassifier(max_depth=20, min_samples_leaf=17,
                                        min_samples_split=16))])
```

As expected, the test results mostly predicted True, indicating that the most reviews were from 2013 and earlier. This is because in the original data, 82.01% of the reviews were True. Therefore, the chances of predicting True were already high. If we predicted all the results as True, we would have an accuracy of 82.01%, which is our worst possible accuracy. 

**The distribution of reviews that done before 2013 and after 2013:**

| index   |   Before 2013 |
|:--------|--------------:|
| **True**    |      **0.820127** |
| False   |      0.179873 |

Overall, train data had accuracy 84.26%, test data of our results had an accuracy of approximately 83.08%, precision of 84.79%, recall of 96.68%, and F-1 score of 90.35%. This suggests that there are more False Positives than False Negatives in the results. The 83.08% accuracy is only about 1.07% higher than the worst possible accuracy, indicating that this is not a good model. We need to include more useful features and perform more feature engineering to create a good final model.

**### Baseline model confusion matrix:**

<iframe src="assets/Baseline Confusion Matrix.html" width=800 height=600 frameBorder=0></iframe>


## Final Model

In our final model, we added two features: `ave_rating` and `submitted_year`.

We believe `ave_rating` is a valuable feature for training our model because when we plotted the mean of `ave_rating` before and after 2013, we found that there was a difference between the two means. This suggests that people's rating habits and preferences may have changed before and after 2013, making `ave_rating` a valid indicator of the time period. 

**Graph 4. The mean of `ave_rating` before and after 2013:**

<iframe src="assets/Mean Ave Rating before and After 2013.html" width=800 height=600 frameBorder=0></iframe>

We can see there is a gap between mean of ave_rating before and after 2013.

Similarly, we believe `submitted_year` is a useful feature because when we plotted the distribution of the number of reviews submitted before and after 2013 for each year, we found that the distributions were different. This indicates that people's rating habits and positivity may have changed before and after 2013, making `submitted_year` a good indicator of the time period.

**Graph 5. The of number of reviews before and after 2013 on recipe submitted in each year:**

<iframe src="assets/Number of reviews before and after 2013 on recipe submitted in each year.html" width=800 height=600 frameBorder=0></iframe>

We can see obviously, there is a huge different between distribution of number of reviews before and after 2013 on recipe submitted in each year.

- Column `ave_rating` is a numerical column, we decide to input it to the model without change.

- Column `submitted_year` is a categorical column, specifically a nominal column, so we decide to input it to the model using OneHotEncoder transform.

- Column `calories` is a numerical column, we decide to input it to the model without change.

- Column `sugar`and `protein` are numerical columns. From the following data we find out there are outliers in both columns, so we decide to input them to the model by QuantileTransformer to reduce the impact of outliers.

 **Mean and maximum of `sugar` and `protein`:**

| index   |    mean |   max |
|:--------|---------|------:|
| sugar   | 63.4709 | 30260 |
| protein | 33.152  |  4356 |

We will have 4 quantitative columns and a nominal column in our final model.

We chose to use the RandomForestClassifier as our final model and obtained the best hyper-parameters: {'Forest__max_depth': 22, 'Forest__n_estimators': 34} using Grid Cross Validation. We selected the RandomForestClassifier because it combines multiple decision trees and outputs the average of all trees' predictions. We believe this ensemble approach will provide more accurate results compared to a single decision tree.

```
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('quantile_sugar',
                                                  QuantileTransformer(n_quantiles=100),
                                                  ['sugar', 'protein']),
                                                 ('submitted_year_onehot',
                                                  OneHotEncoder(),
                                                  ['submitted_year'])])),
                ('Forest',
                 RandomForestClassifier(max_depth=22, n_estimators=34))])
```

Overall, our final model achieved an accuracy of 88.01% on the training data, and the test data resulted in an accuracy of approximately 85.82%, precision of 86.95%, recall of 97.3%, and F-1 score of 91.83%. The test accuracy of final model is 2.74% higher than baseline model, meaning it has a better performance of predicting unseen data, also, it has accuracy more higher than the worst possible accuracy of 82.01%. Therefore, we believe our final model outperforms the baseline model and can better utilize the recipe data to determine whether reviews are from before or after 2013.

**Final model confusion matrix:**

<iframe src="assets/Final Confusion Matrix.html" width=800 height=600 frameBorder=0></iframe>

## Fairness Analysis
To test the fairness of our model, we chose to analyze the accuracy of two groups based on the user ID: odd and even. First, we binarized the `user_id` into odd and even categories. We calculated that the accuracy for the odd user IDs is approximately 85.16%, and for the even user IDs is approximately 86.11%. 

**Accuracy rate of odd & even user ID:**

<iframe src="assets/Accuracy rate of odd & even user ID.html" width=800 height=600 frameBorder=0></iframe>

**Computing parity measures:**

- *C*: Our random forest classifier (1 if the review is before 2013, 0 if the review is after 2013).
- *Y*: Whether the review is truly before 2013 (1) or after 2013 (0).
- *A*: Whether user id is odd or even number (1 if odd, 0 if even).

**Accuracy of *C* in each group:**

| is_odd   |   accuracy |
|:---------|-----------:|
| even     |   0.861092 |
| odd      |   0.851559 |

It seems that the two groups have very close accuracy scores. using permutation test to verify, we have our:

**Null hypothesis**: *H<sub>0</sub>*: the classifier's accuracy is the same for both odd and even user IDs, and any differences are due to chance.

**Alternative hypothesis**: *H<sub>a</sub>*: the classifier's accuracy differs between odd and even user IDs, and the differences are not solely due to chance.

**Test statistic**: Absolute difference in accuracy.

**Significance level**: 0.05.

**Absolute difference in accuracy:**

<iframe src="assets/Absolute difference in accuracy.html" width=800 height=600 frameBorder=0></iframe>

After performing 1000 permutations, we obtained a p-value of 0.207, which is greater than the significance level of 0.05. Therefore, we **fail to reject** the null hypothesis, indicating that we have sufficient evidence to support the accuracy parity between odd and even user IDs are the same. Consequently, *C* is likely to achieve accuracy parity.






