# HS23-FDS
This repository includes the practical assignments of the course Foundations of Data Science.

## Practical 1: Implementation of Linear Regression (`Ridge`, `Lasso`)
Using the [public dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) about wine features, a linear model is trained to predict their quality labels. Furthermore, the model is analyzed with respect to over- and underfitting. As a second step the model is improved by polynomial basis expansion and regularisation. Finally, a closer look at the individual features shows their importance and interaction.

Includes packages: `numpy`, `matplotlib`, `pickle`, `sklearn` : {`StandardScaler`, `PolynomialFeatures`, `Ridge`, `Lasso`, `mean_squared_error`, `cross_val_score`, `LinearRegression`}

## Practical 2: Generative and Discriminative Models
For this task Naïve Bayes Classifier (NBC) and Logistic Regression are compared using several datasets ([iris](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html), [voting](https://archive.ics.uci.edu/ml/datasets/congressional+voting+records), [breast cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer)). For different types of features, different distributions are used:
- the [Gaussian distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html) for continuous features
- the [Bernoulli distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html) for binary features
- the [Multinoulli distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html) for categorical features

Both a NBC and a Logistic Regression model are trained and used for prediction on different dataset sizes. The results are compared to evaluate accuracy.

Includes packages: `numpy`, `matplotlib`, `pickle`, `pandas`, `scipy` : {`norm`, `bernoulli`, `multinomial`}, `sklearn` : {`StandardScaler`, `OrdinalEncoder`, `LogisticRegression`}
