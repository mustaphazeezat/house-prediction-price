# Bayesian House Price Prediction

A Bayesian regression project for predicting house sale prices using **R** and **Stan**. The project compares several Bayesian regression approaches, including linear regression with different priors, ridge regression, and hierarchical models.

## Overview

The project uses housing data containing property characteristics and sale prices. The response variable, `SalePrice`, is square-root transformed before modelling to improve the behaviour of the regression models.

The analysis includes:

* Exploratory data analysis and data cleaning
* Transformation and scaling of numerical variables
* Encoding of categorical variables
* Train/test data splitting
* Bayesian linear regression with flat priors
* Bayesian linear regression with weakly informative priors
* Bayesian ridge regression
* Hierarchical regression using `YearSold`
* Hierarchical regression using `YearBuilt`
* Posterior predictive checks
* MCMC diagnostics and visualisation
* Model comparison using RMSE and R²

## Models

Five Bayesian models are fitted and compared:

1. **Linear Regression with Flat Priors**
2. **Linear Regression with Weakly Informative Priors**
3. **Bayesian Ridge Regression**
4. **Hierarchical Model with Year Sold as a Group Effect**
5. **Hierarchical Model with Year Built as a Group Effect**

The hierarchical models allow the model to account for differences between groups based on the year a property was sold or built.

## Data Preparation

The analysis includes:

* Checking for missing observations
* Inspecting the structure and distribution of the data
* Creating `YearBuilt` categories
* Converting categorical variables to factors
* Scaling selected continuous variables
* Creating dummy variables for categorical predictors
* Splitting the data into 80% training and 20% test sets

The predictors used in the regression models include:

* `LotArea`
* `GarageArea`
* `FullBath`
* `Fireplaces`
* `OverallCond`
* `OverallQual`
* `YearBuilt`
* `SaleCondition`
* `YrSold`

The preprocessing and train/test split are implemented in the R analysis script.

## Bayesian Modelling

The models are implemented using **Stan** through the `rstan` package. Each model estimates posterior distributions for the regression coefficients and generates posterior predictive samples.

For example, the basic Bayesian regression uses a normal likelihood:

```text
Y ~ Normal(α + Xβ, σ)
```

where:

* `α` is the intercept
* `β` represents the regression coefficients
* `σ` is the residual standard deviation
* `X` is the design matrix
* `Y` is the square-root transformed sale price

The project also uses posterior predictive checks to assess how well the fitted models reproduce the observed sale-price distribution.

## Model Evaluation

Models are evaluated on the held-out test set using:

### RMSE

Root Mean Squared Error measures the average magnitude of prediction errors, with larger errors receiving greater weight.

### R²

R² measures the proportion of variation in the observed sale prices explained by the model predictions.

Posterior samples are used to generate predictions, which are then transformed back to the original sale-price scale before calculating the evaluation metrics.

## Technologies

* **R**
* **Stan / rstan**
* **bayesplot**
* **Shinystan**
* **ggplot2**
* **dplyr**
* **caret**
* **MASS**
* **gridExtra**

## Running the Project

### 1. Install the required R packages

```r
install.packages(c(
  "caret",
  "dplyr",
  "rstan",
  "bayesplot",
  "shinystan",
  "ggplot2",
  "gridExtra",
  "MASS"
))
```

### 2. Add the dataset

Place the dataset file in the project directory:

```text
programming_exercise.csv
```

The script loads the data using:

```r
sales_df <- read.csv("programming_exercise.csv", row.names = "Id")
```

### 3. Run the analysis

Run the R script from beginning to end. The script will:

1. Load and prepare the data
2. Perform exploratory analysis
3. Create the training and test datasets
4. Compile the Stan models
5. Run MCMC sampling
6. Perform posterior predictive checks
7. Evaluate and compare the models

## Project Structure

```text
.
├── programming_exercise.csv
├── analysis.R
└── README.md
```

## Results

The final model comparison produces a table containing the **RMSE** and **R²** for each Bayesian model. The model with the lowest test-set RMSE is identified as the best-performing model by the evaluation function.

## Purpose

This project demonstrates the application of **Bayesian statistical modelling** to a real-world regression problem, with particular emphasis on:

* Prior specification
* Hierarchical modelling
* Bayesian regularisation
* Posterior uncertainty
* Posterior predictive checking
* MCMC diagnostics
* Out-of-sample model evaluation
