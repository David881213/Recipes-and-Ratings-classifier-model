# Recipes and Ratings classifier model

By Pin-Hsuan Lai (pilai@ucsd.edu), Pin-Yeh Lai (p2lai@ucsd.edu)

## Framing the Problem

## Baseline Model

## Final Model

## Fairness Analysis
To test the fairness of our model, we chose to analyze the accuracy of two groups based on the user ID: odd and even. First, we binarized the `user_id` into odd and even categories. We calculated that the accuracy for the odd user IDs is approximately 85.16%, and for the even user IDs is approximately 86.11%. These two groups show similar accuracy rates, prompting us to conduct a permutation test for further validation.

Our null hypothesis states that the classifier's accuracy is the same for both odd and even user IDs, and any differences are due to chance. The alternative hypothesis, on the other hand, posits that the classifier's accuracy differs between odd and even user IDs, and the differences are not solely due to chance.

We chose the absolute difference of accuracy as the test statistic and set a significance level of 0.05. After performing 1000 permutations, we obtained a p-value of 0.103, which is greater than the significance level of 0.05. Therefore, we fail to reject the null hypothesis, indicating that we have sufficient evidence to support the accuracy parity between odd and even user IDs are the same. Consequently, our random forest classifier is likely to achieve accuracy parity.


























