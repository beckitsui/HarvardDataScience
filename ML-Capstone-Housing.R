## ----setup, include=FALSE-------------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## -------------------------------------------------------------------------------------------------------
library(caret)
library(dplyr)
library(tidyr)
library(GGally)
library(glmnet)
library(magrittr)
library(MASS)
library(corrplot)

housing_raw <- read.csv("BostonHousing.csv")

housing_raw$index <- seq_along(housing_raw[,1])
set.seed(2023)  # for reproducibility
train_index <- createDataPartition(housing_raw$index, times = 1, p = 0.8, list = FALSE)
train_data <- housing_raw[train_index, ]
test_data <- housing_raw[-train_index, ]


## -------------------------------------------------------------------------------------------------------
summary(train_data)


## -------------------------------------------------------------------------------------------------------
colnames(train_data)
nrow(train_data)


## ----warning=FALSE, message=FALSE-----------------------------------------------------------------------
options(repr.plot.width = 40, repr.plot.height = 40)

train_data |>
  #select(-c('chas', 'b')) |> 
  ggpairs(
        mapping = ggplot2::aes(size = 0.005),
        lower = list(continuous = wrap("points", alpha = 0.3, size=0.1, color = 'blue')), 
        diag = list(continuous = wrap("barDiag", alpha = 0.3)),  
        font.label = list(size = 4, face = "bold"),
        ggtheme = theme_minimal()
       )


## -------------------------------------------------------------------------------------------------------
# correlation matrix plotting
cor_plot_matrix <- 
  cor(train_data[,c("crim","zn","indus", "chas", "nox","rm","age","dis","rad","tax",
                    "ptratio","lstat","medv")])

corrplot(cor_plot_matrix, method = 'color', type='upper', 
         order = 'hclust',tl.col = "black", tl.srt = 45)


## -------------------------------------------------------------------------------------------------------
#view the correlation values 
cor_plot_matrix


## -------------------------------------------------------------------------------------------------------
RMSE <- function(true, predict) {
  differences <- true - predict
  differences <- differences[!is.na(differences)]
  sqrt(mean(differences^2))
}


## -------------------------------------------------------------------------------------------------------
R2 <- function(true, predict){
  #Total Sum of Squares (TSS)
  tss <- sum((true - mean(true))^2)
  #Residual Sum of Squares (RSS)
  rss <- sum((true - predict)^2)
  1 - (rss / tss)
}


## -------------------------------------------------------------------------------------------------------
#create model "cv_model_base" with all original features
train_data_base <- data.frame(scale(train_data[c("crim","zn","indus", "chas", "nox","rm",
                                                 "age","dis","rad","tax","ptratio","lstat")]))
train_data_base$medv <- train_data$medv

#prepare a matrix for glmnet
x_train_base <- model.matrix(medv ~ crim + zn + indus + chas + nox + rm + age + dis + 
                               rad + tax + ptratio + lstat - 1, train_data_base)
y_train_base <- train_data$medv

#use cross validation to find the optimal lambda 
cv_model_base <- cv.glmnet(x_train_base, y_train_base, alpha=1)

lambda_base <- cv_model_base$lambda.min


## -------------------------------------------------------------------------------------------------------
test_data_base <- data.frame(scale(test_data[c("crim","zn","indus", "chas", "nox","rm",
                                               "age","dis","rad","tax","ptratio","lstat")]))
test_data_base$medv <- test_data$medv

x_test_base <- model.matrix(medv ~ crim + zn + indus + chas + nox + rm + age + dis + 
                              rad + tax + ptratio + lstat -1, test_data_base)

predict_base <- predict(cv_model_base, s=lambda_base, newx = x_test_base)

y_test <- test_data$medv


## -------------------------------------------------------------------------------------------------------
RMSE(y_test, predict_base)
R2(y_test, predict_base)


## -------------------------------------------------------------------------------------------------------
library(dplyr)
cor_plot_ranked <- data.frame(cor_plot_matrix) 
cor_plot_ranked['medv']|>
  arrange(desc(abs(medv))) 


## -------------------------------------------------------------------------------------------------------
transformations <- list(
  og = function(x) x,
  log = function(x) {ifelse(x > 0, log(x+1), x)},  
  sqrt = function(x) sqrt(x),
  square = function(x) x^2
)


## -------------------------------------------------------------------------------------------------------
library(glmnet)

test_transformation <- function(data, feature, transformations, response = "medv") {
  results <- list()
  for (trans_name in names(transformations)) {
    # apply transformation for feature
    data[[paste0(feature, "_", trans_name)]] <- 
      transformations[[trans_name]](data[[feature]])
    
    # prepare data for LASSO
    x <- model.matrix(reformulate(termlabels = paste0(feature, "_", trans_name), 
                                  response = response), data)
    y <- data[[response]]
    
    # fit LASSO model
    cv_model <- cv.glmnet(x, y, alpha = 1)
    
    # store results
    results[[trans_name]] <- min(cv_model$cvm)  # store min. cross-validation error
  }
  return(results)
}



## -------------------------------------------------------------------------------------------------------
features <- c("crim", "zn", "indus", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", 
              "lstat") #"chas" is removed

all_results <- list()

for (feature in features) {
  all_results[[feature]] <- test_transformation(train_data, feature, transformations)
}
all_results


## -------------------------------------------------------------------------------------------------------
transforming_train <- train_data |>
  mutate(log_crim = log(crim + 1),
          log_zn = log(zn + 1),
          log_indus = log(indus + 1),
          sq_rm = rm^2,
          sq_age = age^2,
          log_dis = log(dis+1),
          log_lstat = log(lstat + 1))
  
transformed_train_data <- transforming_train[c('log_crim','log_zn','log_indus','sq_rm',
                                               'sq_age','log_dis','log_lstat','chas',
                                               'tax','ptratio','nox','rad')] |>
  scale() |>
  data.frame()

transformed_train_data$medv = train_data$medv
head(transformed_train_data)


## -------------------------------------------------------------------------------------------------------
#prepare a matrix for glmnet
x_train_transformed <- model.matrix(medv ~ log_crim + log_zn + log_indus + sq_rm + 
                                      sq_age + log_dis + log_lstat + rad + chas + 
                                      tax + ptratio + nox- 1, 
                                    transformed_train_data)
y_train <- train_data$medv

#use cross validation to find the optimal lambda 
cv_model_transformed <- cv.glmnet(x_train_transformed, y_train, alpha=1)
lambda_transformed <- cv_model_transformed$lambda.min


## -------------------------------------------------------------------------------------------------------
#prepare test set for prediction
transforming_test <- test_data |>
  mutate(log_crim = log(crim + 1),
          log_zn = log(zn + 1),
          log_indus = log(indus + 1),
          sq_rm = rm^2,
          sq_age = age^2,
          log_dis = log(dis+1),
          log_lstat = log(lstat + 1))
  
transformed_test_data <- transforming_test[c('log_crim','log_zn','log_indus','sq_rm','sq_age','log_dis','log_lstat','chas','tax',
                                             'ptratio','nox','rad')] |>
  scale() |>
  data.frame()

transformed_test_data$medv <- test_data$medv

x_test_transformed <- model.matrix(medv ~ log_crim + log_zn + log_indus + sq_rm + sq_age + log_dis + 
                                     log_lstat + rad + chas + tax + ptratio + nox - 1, 
                                   transformed_test_data)

predict_LASSO_transformed <- predict(cv_model_transformed,
                               newx = x_test_transformed)

y_test <- test_data$medv


## -------------------------------------------------------------------------------------------------------
RMSE(y_test, predict_LASSO_transformed)
R2(y_test, predict_LASSO_transformed)


## -------------------------------------------------------------------------------------------------------
library(randomForest)

select_features_rf <- c("crim","zn","indus", "chas", "nox","rm","age","dis","rad","tax",
                        "ptratio","lstat", "medv")

train_data_rf <- train_data[select_features_rf]

# create cross validation parameter / train control
train_control <- trainControl(method = 'cv', number = 5)

# create tuning grid
tune_grid <- expand.grid(mtry = c(2, 3, 4, 5, 6))

# train Random Forest model with tuned parameters
rf_model_tuned <- train(medv ~ ., 
                        data = train_data_rf,
                        method = 'rf',
                        trControl = train_control,
                        tuneGrid = tune_grid)

rf_model_tuned$bestTune$mtry

mtry <- rf_model_tuned$bestTune$mtry


## -------------------------------------------------------------------------------------------------------
# build Random Forest model with original data
rf_model <- randomForest(medv ~ ., 
                         data = train_data_rf, 
                         ntree = 500, 
                         mtry = mtry)

# view model information 
print(rf_model)
importance(rf_model)
# plotting feature importance
varImpPlot(rf_model) 


## -------------------------------------------------------------------------------------------------------
# test Random Forest model by predicting test set
test_data_rf <- test_data[select_features_rf]
predictions_rf <- predict(rf_model, test_data_rf)

# get RMSE of the Random Forest model 
rf_result <- data.frame(medv_actual = test_data_rf$medv,
                        medv_predicted = predictions_rf
                        )

RMSE(rf_result$medv_actual, rf_result$medv_predicted)
R2(rf_result$medv_actual, rf_result$medv_predicted)


## -------------------------------------------------------------------------------------------------------
ensemble_predictions <- (predictions_rf + predict_LASSO_transformed) / 2

RMSE(test_data_rf$medv, ensemble_predictions)
R2(test_data_rf$medv, ensemble_predictions)

