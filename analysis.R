library(caret)
library(dplyr)
library(rstan)
library(bayesplot)
library(shinystan)
library(ggplot2)
library(gridExtra)
library(MASS)

set.seed(123) 


options(mc.cores = 1) #set number of CPU cores to use



### Reading and cleaning of data
sales_df <- read.csv("programming_exercise.csv", row.names = "Id")

View(sales_df)
summary(sales_df$SalePrice)


#Check for missing data
sum(is.na(sales_df))


#Check structure of the data frame
str(sales_df)

#Create categories for 'YearBuilt' between 1872 and 2010
sales_df$YearBuilt_cat <- cut(sales_df$YearBuilt,
                               breaks = c(1871, 1900, 1950, 2000, 2010),
                               labels = c("1872-1900", "1901-1950", "1951-2000", "2001-2010"),
                               include.lowest = TRUE, right = TRUE)
unique(sales_df$YearBuilt_cat)

### Transform variables to appropriate type.
sales_df$YrSold <- as.factor(sales_df$YrSold)
sales_df$SaleCondition <- as.factor(sales_df$SaleCondition)
sales_df$YearBuilt_cat <- as.factor(sales_df$YearBuilt_cat)

### exploratory data analysis ###

#Boxplot for Sales price 
boxplot(sales_df$SalePrice, main="Boxplot of Sale price", horizontal=TRUE)

#Histogram of Sales price to check for heavy tails
hist(sales_df$SalePrice, breaks=30, main="Histogram of Sale price", xlab="Sale price")

#Q-Q plot to comparing distribution of Sales price to a normal distribution
qqnorm(sales_df$SalePrice)
qqline(sales_df$SalePrice, col="red")


#categorical variables
categorical_vars <- names(sales_df)[sapply(sales_df, is.factor)]
categorical_vars

#boxplots for each categorical variable
boxplots <- lapply(categorical_vars, function(cat_var) {
  ggplot(sales_df, aes_string(x = cat_var, y = "SalePrice")) +
    geom_boxplot(outlier.colour = "red", fill = "lightblue") +
    theme_minimal() +
    labs(title = paste("Boxplot of SalePrice by", cat_var), x = cat_var, y = "SalePrice") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
})

#boxplots in a grid
grid.arrange(grobs = boxplots, ncol = 2) 

#continuous variables (excluding sales price)
continuous_vars <- names(sales_df)[sapply(sales_df, is.numeric) & names(sales_df) != "SalePrice"]

#scatter plots for each continuous variable
scatter_plots <- lapply(continuous_vars, function(cont_var) {
  ggplot(sales_df, aes_string(x = cont_var, y = "SalePrice")) +
    geom_point(color = "blue", alpha = 0.6) +
    geom_smooth(method = "lm", se = FALSE, color = "red") +  
    theme_minimal() +
    labs(title = paste("Scatter Plot of SalePrice vs", cont_var), x = cont_var, y = "SalePrice")
})

#scatter plots in a grid
grid.arrange(grobs = scatter_plots, ncol = 2) 


#Boxplot for OverallCond vs SalePrice
ggplot(sales_df, aes(x = factor(OverallCond), y = SalePrice)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Boxplot of SalePrice by OverallCond", x = "Overall Condition", y = "SalePrice")

#Boxplot for OverallQual vs SalePrice
ggplot(sales_df, aes(x = factor(OverallQual), y = SalePrice)) +
  geom_boxplot(fill = "lightgreen", outlier.color = "red", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Boxplot of SalePrice by OverallQual", x = "Overall Quality", y = "SalePrice")


#scale continous varibales to ensure interpretability in the model and convergence
df <- sales_df 
df$LotArea <- scale(df$LotArea)
df$GarageArea <- scale(df$GarageArea)
df$FullBath <- scale(df$FullBath)
df$Fireplaces<- scale(df$Fireplaces)

#Create train-test split
train_index <- createDataPartition(df$SalePrice, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

#Encoding of categorical variables and creating base level category
X_YearBuilt_cat <- model.matrix(~ YearBuilt_cat - 1, data = train_data)[, -1]
X_SaleCondition <- model.matrix(~ SaleCondition - 1, data = train_data)[, -1]
X_YrSold <- model.matrix(~ YrSold - 1, data = train_data)[, -1]

#Create design matriix for train data
X <- cbind(
  train_data$LotArea, 
  train_data$GarageArea, 
  train_data$FullBath,
  train_data$Fireplaces,
  train_data$OverallCond,
  train_data$OverallQual,
  X_YearBuilt_cat, 
  X_SaleCondition,
  X_YrSold
)

#Add column names to X
colnames(X) <- c(
  "LotArea", "GarageArea", "FullBath", "Fireplaces", "OverallCond", "OverallQual",
  colnames(X_YearBuilt_cat), colnames(X_SaleCondition), colnames(X_YrSold)
)
#Create design matriix for test data
Xt_YearBuilt_cat <- model.matrix(~ YearBuilt_cat - 1, data = test_data)[, -1]
Xt_SaleCondition <- model.matrix(~ SaleCondition - 1, data = test_data)[, -1]
Xt_YrSold <- model.matrix(~ YrSold - 1, data = test_data)[, -1]

Xt <- cbind(
  test_data$LotArea, 
  test_data$GarageArea, 
  test_data$FullBath,
  test_data$Fireplaces,
  test_data$OverallCond,
  test_data$OverallQual,
  Xt_YearBuilt_cat, 
  Xt_SaleCondition,
  Xt_YrSold
)

### Stan data
stan_data_linearregression_sqrt <- list(
  N = nrow(train_data),
  K = ncol(X),   
  X = X,         
  Y = as.vector(sqrt(train_data$SalePrice))
)


stan_data_hierarchical_yearSold_sqrt <- list(
  N = nrow(train_data),
  K = ncol(X),
  X = X,
  Y = as.vector(sqrt(train_data$SalePrice)), 
  group = as.integer(train_data$YrSold),       
  G = length(unique(train_data$YrSold))  
)

stan_data_hierarchical_yearBuilt_sqrt <- list(
  N = nrow(train_data),
  K = ncol(X),
  X = X,
  Y = as.vector(sqrt(train_data$SalePrice)), 
  group = as.integer(train_data$YearBuilt_cat),       
  G = length(unique(train_data$YearBuilt_cat))  
)

### Models ###

### Linear Model with flat priors ###

linearRegresion_flatPrior.code = '
  data {
    int<lower=0> N;           // Number of data points
    int<lower=0> K;           // Number of predictors 
    matrix[N, K] X;           // Design matrix
    vector[N] Y;              // Response variable
  }
  
  parameters {
    real alpha;               // Intercept
    vector[K] beta;           // Coefficients for predictors
    real<lower=0> sigma;      // Error standard deviation
  }
  
  model {
    // Flat priors (non-informative priors)
    alpha ~ normal(0, 1e6);     // Intercept (very wide prior)
    beta ~ normal(0, 1e6);      // Coefficients (very wide prior)
    sigma ~ exponential(1e-6);  // Error standard deviation (very wide prior)
  
    // Likelihood
    Y ~ normal(alpha + X * beta, sigma);  // Linear model
  }
  generated quantities{
    vector[N] y_rep;
  
    for(n in 1:N){
      y_rep[n] = normal_rng(alpha + dot_product(X[n], beta), sigma);
    }
  }'

# Compile the Stan model
linearRegresion_flatPrior.stan_model <- stan_model(model_code = linearRegresion_flatPrior.code,
                                                   model_name="linearRegresion_flatPrior.stan_model")


# Fit the model 
fit_lr_fprior <- sampling(linearRegresion_flatPrior.stan_model,
                          data = stan_data_linearregression_sqrt, 
                          iter = 4000, warmup = 2000,  chains = 4)

mcmc_trace(fit_lr_fprior)
mcmc_dens(fit_lr_fprior)

launch_shinystan(fit_lr_fprior)

### Posterior Predictive Check (PPC)
posterior_predict_flatPrior <- (extract(fit_lr_fprior, pars = "y_rep")[[1]])^2

ppc_hist(train_data$SalePrice, posterior_predict_flatPrior[sample(NROW(posterior_predict_flatPrior), 11), ])


### Linear Model with weakly informative priors ###

linearRegresion_weakPrior.code <-'

  data {
    int<lower=0> N;           // Number of data points
    int<lower=0> K;           // Number of predictors 
    matrix[N, K] X;           // Design matrix
    vector[N] Y;              // Response variable
  }
  parameters{
    real alpha;               // Intercept
    vector[K] beta;           // Coefficients for predictors
    real<lower=0> sigma;      // Error standard deviation
  }
  
  model{
   
  alpha ~ normal(0, 10);        // Prior for intercept
  beta ~ normal(0, 10);
  sigma ~  normal(0, 10);  
  
  Y ~ normal(alpha + X * beta, sigma);  // Linear model
  }
  generated quantities{
    vector[N] y_rep;
  
    for(n in 1:N){
      y_rep[n] = normal_rng(alpha + dot_product(X[n], beta), sigma);
  }}'

# Compile the Stan model
linearRegresion_weakPrior.stan_model <- stan_model(model_code = linearRegresion_weakPrior.code,
                                                   model_name="linearRegresion_weakPrior.stan_model")


# Fit the model to data
fit_lr_wprior <- sampling(linearRegresion_weakPrior.stan_model,
                          data = stan_data_linearregression_sqrt, 
                          iter = 4000, warmup = 2000,  chains = 4)

mcmc_trace(fit_lr_wprior)
mcmc_dens(fit_lr_wprior)

launch_shinystan(fit_lr_wprior)

### Posterior Predictive Check (PPC)

posterior_predict_wPrior <- (extract(fit_lr_wprior, pars = "y_rep")[[1]])^2 
ppc_hist(train_data$SalePrice, posterior_predict_wPrior[sample(NROW(posterior_predict_wPrior), 11), ])

## Ridge regression model with weak informative prior##

linearRegresion_ridge.code <-'
  data {
    int<lower=0> N;               // Number of data points
    int<lower=0> K;               // Number of predictors (including dummy variables)
    matrix[N, K] X;               // Design matrix
    vector[N] Y;                  // Response variable
  }
  parameters {
    real alpha;                   // Intercept
    vector[K] beta;               // Coefficients for predictors (including dummy variables)
     real<lower=0> lambda;         // Ridge regularization parameter (penalty)
    real<lower=0> sigma;          // Error standard deviation
    
  }
  model {
    alpha ~ normal(0, 10);        // Informative prior for intercept
   for (k in 1:K) {
     beta[k] ~ normal(0, sqrt(lambda));
   }
    
    // Specify prior for lambda (weakly informative prior)
  lambda ~ normal(0, 0.5) T[0, ];      // Prior for lambda (weakly informative)
  
    sigma ~ normal(0, 10);  // Standard deviation prior for error term
    
    // Likelihood function (linear regression model)
    Y ~ normal(alpha + X * beta, sigma); // Linear model
  }
  
  generated quantities{
    vector[N] y_rep;
  
    for(n in 1:N){
      y_rep[n] = normal_rng(alpha + dot_product(X[n], beta), sigma);
  }}'

# Compile the Stan model
linearRegresion_ridge.stan_model <- stan_model(model_code = linearRegresion_ridge.code,
                                               model_name="linearRegresion_ridge.stan_model")


# Fit the model 
fit_lr_ridge <- sampling(linearRegresion_ridge.stan_model, 
                         data = stan_data_linearregression_sqrt, 
                         iter = 4000, warmup = 2000,  chains = 4, 
                         control = list(adapt_delta = 0.99))

mcmc_trace(fit_lr_ridge)
mcmc_dens(fit_lr_ridge)

launch_shinystan(fit_lr_ridge)

### Posterior Predictive Check (PPC)
posterior_predict_lr_ridge <- (extract(fit_lr_ridge, pars = "y_rep")[[1]])^2 
ppc_hist(train_data$SalePrice, posterior_predict_lr_ridge[sample(NROW(posterior_predict_lr_ridge), 11), ])

##  Hierarchical model model with weak informative prior using year sold as random effect##

hierarchical_yearSold.code <- 'data {
  int<lower=0> N;          // number of data points
  int<lower=0> K;          // number of predictors (including dummies)
  int<lower=0> G;          // Number of groups
  matrix[N, K] X;          // Design matrix
  int<lower=1, upper=G> group[N]; // Group indicator for hierarchical model
  vector[N] Y;             // Response variable
  
}
parameters {
  real alpha;              // Intercept
  vector[K] beta;          // Coefficients for predictors
  real<lower=0> sigma;     // Error standard deviation
  real mu_group[G];        // Group-level intercepts (mean SalePrice per group)
  real<lower=0> sigma_group; // Standard deviation for group-level intercepts
}
model {
  alpha ~ normal(0, 10);   // Prior for intercept
  beta ~ normal(0, 5);     // Prior for coefficients
  sigma ~ normal(0, 5) T[0, ];   // Prior for sigma
  mu_group ~ normal(0, 1); // Prior for group-level means
  sigma_group ~ normal(0, 1) T[0, ];  // Prior for group-level sigma
  
  // Group-level intercepts
  for (n in 1:N) {
    Y[n] ~ normal(alpha + X[n] * beta + mu_group[group[n]], sigma);
  }}
  generated quantities{
    vector[N] y_rep;
  
    for(n in 1:N){
      y_rep[n] = normal_rng(alpha + dot_product(X[n], beta) + mu_group[group[n]], sigma);
  }}'

# Compile the Stan model
hierarchical_yearSold.stan_model <- stan_model(model_code =hierarchical_yearSold.code,
                                      model_name="hierarchical_yersSold.stan_model")


# Fit the model to data
fit_hierarchical_yearSold <- sampling(hierarchical_yearSold.stan_model, 
                             data = stan_data_hierarchical_yearSold_sqrt, 
                             iter = 4000, warmup = 2000,  chains = 4, 
                             control = list(adapt_delta = 0.99))

mcmc_trace(fit_hierarchical_yearSold)
mcmc_dens(fit_hierarchical_yearSold)

launch_shinystan(fit_hierarchical_yearSold) 

### Posterior Predictive Check (PPC)
posterior_predict_hierarchical_yearSold <- (extract(fit_hierarchical_yearSold, pars = "y_rep")[[1]])^2 
ppc_hist(train_data$SalePrice, posterior_predict_hierarchical_yearSold[sample(NROW(posterior_predict_hierarchical_yearSold), 11), ])

##  Hierarchical model with weak informative prior using year built as random effect##

# Compile the Stan model
hierarchical_yearBuilt.stan_model <- stan_model(model_code = hierarchical_yearSold.code,
                                               model_name="hierarchical_yearBuilt.stan_model")

# Fit the model to data
fit_hierarchical_yearBuilt <- sampling(hierarchical_yearBuilt.stan_model, 
                                      data = stan_data_hierarchical_yearBuilt_sqrt, 
                                      iter = 4000, warmup = 2000,  chains = 4, 
                                      control = list(adapt_delta = 0.99))



mcmc_trace(fit_hierarchical_yearBuilt)
mcmc_dens(fit_hierarchical_yearBuilt)

launch_shinystan(fit_hierarchical_yearBuilt)

### Posterior Predictive Check (PPC)
posterior_predict_hierarchical_yearBuilt <- (extract(fit_hierarchical_yearBuilt, pars = "y_rep")[[1]])^2 
ppc_hist(train_data$SalePrice, posterior_predict_hierarchical_yearBuilt[sample(NROW(posterior_predict_hierarchical_yearBuilt), 11), ])




#### Model evaluation ####

#Function to evaluate model with RMSE and Rsquare
evaluate_models <- function(models, Xt, test_salePrice, metrics = c("rmse", "r_squared")) {
  #Function to compute RMSE
  compute_rmse <- function(predictions, actual) {
    sqrt(mean((predictions - actual)^2))
  }
  
  #Function to compute R-squared
  compute_r_squared <- function(predictions, actual) {
    ss_total <- sum((actual - mean(actual))^2)
    ss_residual <- sum((actual - predictions)^2)
    r_squared <- 1 - (ss_residual / ss_total)
    return(r_squared)
  }
  
  # Initialize results list
  model_results <- list()
  metrics_table <- data.frame(Model = character(), RMSE = numeric(), R2 = numeric(), stringsAsFactors = FALSE)
  
  # Loop through each model
  for (model_name in names(models)) {
    model <- models[[model_name]]
    
    # Extract posterior samples
    posterior_samples <- as.data.frame(rstan::extract(model)) 
    alpha <- posterior_samples$alpha
    beta <- as.matrix(posterior_samples[ , grep("^beta", names(posterior_samples))])
    
    # Check dimensions
    if (ncol(beta) != ncol(Xt)) {
      stop(paste0("Mismatch between beta dimensions and Xt for model: ", model_name))
    }
    
    # Generate posterior predictions
    pred_posterior <- Xt %*% t(beta) + matrix(alpha, nrow = nrow(Xt), ncol = length(alpha), byrow = TRUE)
    
    # Calculate the mean prediction
    pred_mean <- rowMeans(pred_posterior)
    
    # Square the predictions to convert back to the original scale
    pred_mean_original_scale <- pred_mean^2
    
    # Calculate metrics
    rmse <- if ("rmse" %in% metrics) compute_rmse(pred_mean_original_scale, test_salePrice) else NA
    r_squared <- if ("r_squared" %in% metrics) compute_r_squared(pred_mean_original_scale, test_salePrice) else NA
    
    # Store results for the model
    model_results[[model_name]] <- list(rmse = rmse, r_squared = r_squared, predictions = pred_mean_original_scale)
    
    # Add metrics to the table
    metrics_table <- rbind(metrics_table, data.frame(Model = model_name, RMSE = rmse, R2 = r_squared))
  }
  
  # Identify the best model based on RMSE (lowest RMSE)
  best_model_name <- names(model_results)[which.min(sapply(model_results, function(x) x$rmse))]
  best_model <- model_results[[best_model_name]]
  
  # Return the results, best model, and metrics table
  return(list(results = model_results, best_model = best_model, best_model_name = best_model_name, metrics = metrics_table))
}



## List of models
models <- list(
  fit_lr_fprior = fit_lr_fprior,
  fit_lr_wprior = fit_lr_wprior,
  fit_lr_ridge = fit_lr_ridge,
  fit_hierarchical_yearSold = fit_hierarchical_yearSold,
  fit_hierarchical_yearBuilt = fit_hierarchical_yearBuilt
)

## Result based on RMSE
results <- evaluate_models(models, Xt, test_data$SalePrice)
sapply(results$results, function(x) x$rmse)
results$metrics
results$best_model
results$best_model_name
