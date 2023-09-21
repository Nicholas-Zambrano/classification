Classification readme:

First of all, I only completed the spam filtering part 1 task. I followed the naive bayes algorithm guide in the data file. My code consists of five functions, with four of them containing the algorithm for filtering the emails.

Estimate_log_class_priors:

This function is calculating the log class priors, which is the prior probability of that particular class (ham or spam) occurring. In this data set, the label value 0 represents ham and the label value 1 represent spam.  Hence, started to count the number of times ham or spam appeared in the data, this was done with the help of the function ‘np.count_nonzero(0)’. Then I calculated the total number of samples in the dataset, as this will be used as the denominator when finding the proportion of ham and spam. The proportions of those emails were saved in variables called ‘proportion_ham’ and ‘proportion_spam’. Thus, I calculated the logarithm for each of the proportion variables, which was then stored in a NumPy array called ‘self.log_class_priors’.

Estimate_log_class_conditional_likelihoods:

This function estimates the log class conditional likelihoods of each feature for that specified class. I began initializing the variables which will later be updated with new values, such as setting spam and ham count to 0, as well as creating Boolean masks that checks whether each label is equal to 0 or 1. Hence, I created an array which will contain the rows that correspond to the ham class and another array that corresponds to the spam class. Then, I initialised a theta array where it contains rows that corresponds to a class and columns that correspond to a feature. The values inside the array, will be the estimated log class conditional likelihoods of each feature for each class.

Furthermore, I iterated through each ham and spam class to collect the number of messages for that class and the total word count for each feature, these values are stored in variables called ‘ham_count’ and ‘spam_count’. Therefore, I computed the log class conditional likelihoods for each feature with the formula ‘np.log ((word_count + alpha) / (n_c + alpha * n_features))’. As a result, these values are stored in the theta array.

Train:

This function trains the Naïve bayes classifier for that given data, as one of the parameters is ‘train_data’ which contains the samples and features and the other was ‘train_labels’ which is used to specify the class labels in the training data.  Hence, I called the function ‘self.estimate_log_class_priors’ in order to estimate the prior probabilities of each class, as well as calling the function ‘self.estimate_log_class_conditional_likelihoods’ to estimate the log conditional likelihoods of each feature. These values are essential for calculating the posterior probabilities during the predict function.



Predict:

The predict function has the parameter ‘test_data’ which is the matrix containing the feature values and the samples. This function calculates the posterior probabilities of each class for each sample, which is done by calling the ‘log_class_priors’ used in the ‘train function. 

Furthermore, the number of samples is determined and stored in a matrix full of zeros, so that the classifier can eventually iterate over each sample and predict its class label, which will be stored in that matrix. Then we call the ‘log_priors_function’ used in the ‘train function’, which is assigned to variables corresponding with their class label.  Next, I iterated through the test samples in order to calculate the log conditional likelihoods for each feature using the ‘theta’ function, this was also used in the ‘train’ function. Therefore, I computed the product of the feature values and the log conditional likelihoods and then summed it up.  As a result, the posterior probability is calculated by adding the log priors and summed value.

Lastly, I returned predicted class as an array, this was determined based on which posterior probability is greater. As a result, my accuracy predictions on the test data is 0.836 which means that my naive bayes classifier correctly predicted the class labels of approximately 83.6%.

