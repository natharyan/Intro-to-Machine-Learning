
- if feature categorical - use mode.
- knn imputation - for both categorical and numerical variables

- do knn imputation, pca

- chi-square, categorical

- Linear Regression: Error = Actual Value - Predicted Value
	- Best Regression model minimizes the sum of squared errors. - only one line that does this.
	- r-squared is better than just taking sum of squared errors as its evaluation is independent of number of training points.
	- Training outliers can make huge difference.
	- Error or residual is just the difference of actual and predicted value

- Unsupervised Learning: Flags are not given in the dataset.

- clustering unlabeled data
- cluster centering - just see closer to which set of points - use rubberbands with minimum energy - minimizes r-square distance.
	- cluster identification - perp. bisector of line joining cluster centres.

- feature scaling - span feature values to a comparable range - otherwise larger feature values will always dominate in comparison.
- scaling formula: $x' = \frac{x - x_{min}}{x_{max} - x_{min}}$
- numpy array - array of data points - each data point array of feature values.

- Feature Selection - get minimal best features to form training and testing data.
- Anyone can make mistakes--be skeptical of your results!
- 100% accuracy should generally make you suspicious. Extraordinary claims require extraordinary proof.
- If there's a feature that tracks your labels a little too closely, it's very likely a bug!
- If you're sure it's not a bug, you probably don't need machine learning--you can just use that feature alone to assign labels.
- SelectPercentile and SelectKBest - univariate sklearn feature selection.


- Bias - does not learn well from the data - oversimplified training. - **high error on training set.** - **Using few features in training.**
- Variance - prediction too dependent on training data - overfitting. - **high error on test set.** - while training on the data - train it to accurately fit just the training data.

- underfit/high bias $\leftarrow \text{{no. of features}} \rightarrow$ overfitting/high variance. - find optimal using Regularization.

- smaller sum of squares error for greater number of features.
- overfit decision tree - small training set and lots of features.

- TFIDF:
	- TF - word frequency within the document.
	- IDF - word frequency across all documents.

- One dimensional dataset - along straight line (with insignificant noise) - line indicates a feature which is new or originally present.
- PCA just does shifts and rotations to x-y axis so a quadratic graph is 2 dimensional in PCA.
- PCA - moves new center to center of data points - x-axis along most data points(axis of most variation - least intertia means most variance. variance - can take larger number of values), other axis orthogonal to x-axis.
- Latent features - compress large number of features to a smaller meaningful set of features conveying same information with regards to outcome.
- maximal principal components = number of input features.
- Do PCA when:
	- deduce latent features
	- dimensionality reduction
		- visualise high dimensional data
		- reduce noise
		- make classification and regression models work better because lesser features(inputs).
- Train/Test split -> PCA(fit_transform on training data - testing data will test on this so don't fit PCA on it. Just transform on testing features after fitting of classification algorithm is done.) -> classification algorithm.
- K-fold Cross validation - k times running with different train/test splits and taking average of those results.
- in K-Fold make sure to shuffle - in order may end up training on just one type of label and testing on the other. Can use shuffle in KFold or StratifiedKFold.

- Recall - probability - correctly predict Hugo Chavez given person is actually Hugo Chavez - $point/row$  - minimize false negatives - $\textit{true positives/true positives + false negatives}$ = $\textit{true positives/total samples that should have been positive}$
- Precision - probability - if predicted as Hugo Chavez then actually Hugo Chavez - $point/column$ - minimize false positives - $\textit{true positives/true positives + false positives}$$ = $\textit{true positives/total positives}$.
- True positive - positive for correct prediction.
- False positive - positive for when should be negative - all except data point in column.
- False negative - negative when should be positive - all except data point in row of confusion matrix.

- $F_1$ score - harmonic mean of precision and recall. Use this when in doubt.

- ==First features in features_list is the target==.

- A machine learning algorithm is good when it gives true positives for the less common of the two labels.

1. Dataset or question you want to work on
2. Feature selection
3. Algorithm
4. Validation