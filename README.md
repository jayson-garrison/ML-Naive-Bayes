# ML-Naive-Bayes

## Assignment

CS-5333 Project 1

## Project Description

With two different data sets, cosisting of image data and word data, apply the Naive Bayes Classification to learn the associated labels. Utilize two different models, the multivariate Bernoulli event and multinomial event models for posterior probability calculations. Analyze the effect of different k-values in the application of Laplace Smoothing for both models on both data sets. Determine training and testing accuracies and verify using five fold cross validation. Report the results.

## Data Sets Used

1. [Images](https://www.kaggle.com/c/digit-recognizer/data?select=train.csv)
2. [Words](https://www.kaggle.com/balaka18/email-spam-classification-dataset-csv)


## Interesting Points to Consider

- The multinomial model can be applied on the image data set if intensities are known for each pixel
- Laplace constant of 0 < k < 1 improves accuracy for multivariate Bernoulli event model
- In general, the algorithmic complexity of the classify() method is not optimal

## What I have learned

### Languages

I have learned to implement this machine learning algorithm in Python. By doing this, I have become more comfortable and compitent in developing significant projects in Python.

### Development

- Linux based development
- Anaconda to manage several packages

### Packages

- NumPy

### Insight

- Manage and organize large data sets
- Implement abstract classes and methods
- Generalize a modular model implementation to support both data sets
- Learned the application of Bayesian statistics for machine learning algorithms
- Analyzed the impact of algorithmic efficiency in the context of large data processing
- Utilized NumPy to handle large arrays of data

