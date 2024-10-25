#START
##############################################################################################################
############################################################################################################
install.packages("caret")
install.packages("glmnet")
install.packages("glm2")
install.packages("dplyr")
install.packages("MASS")
install.packages("ROCR")
install.packages("pROC")


library(glm2)
library(readxl)
library(purrr)
library(dplyr)
library(caret)
library(ROCR)
library(pROC)
library(glmnet)
library(MASS)


# Datasets
jnj_returns <- read_excel("/Users/ghadielhayek/Desktop/Machine Learning/JNJ Return Dependant Variable.xlsx")
gs_returns <- read_excel("/Users/ghadielhayek/Desktop/Machine Learning/GS Returns Independant.xlsx")
mcd_returns <- read_excel("/Users/ghadielhayek/Desktop/Machine Learning/MCD Returns Independant.xlsx")
mmm_returns <- read_excel("/Users/ghadielhayek/Desktop/Machine Learning/MMM Returns Independant.xlsx")
pg_returns <- read_excel("/Users/ghadielhayek/Desktop/Machine Learning/PG Return Independant.xlsx")


#Lagged independant variables by 1 day
gs_returns_lagged <- gs_returns %>%
  mutate(Lagged_GS_Return = lag(Return, 1))

mcd_returns_lagged <- mcd_returns %>%
  mutate(Lagged_MCD_Return = lag(Return, 1))

mmm_returns_lagged <- mmm_returns %>%
  mutate(Lagged_MMM_Return = lag(Returns, 1))

pg_returns_lagged <- pg_returns %>%
  mutate(Lagged_PG_Return = lag(Return, 1))

gs_returns_lagged <- gs_returns_lagged %>%
  rename(gs_lagged_return = Lagged_GS_Return)

mcd_returns_lagged <- mcd_returns_lagged %>%
  rename(mcd_lagged_return = Lagged_MCD_Return)

mmm_returns_lagged <- mmm_returns_lagged %>%
  rename(mmm_lagged_return = Lagged_MMM_Return)

pg_returns_lagged <- pg_returns_lagged %>%
  rename(pg_lagged_return = Lagged_PG_Return)

final_data <- final_data %>%
  rename(
    jnj_close = Close.x,
    jnj_return = Return.x,
    gs_lagged_return = Lagged_GS_Return,
    mcd_lagged_return = Lagged_MCD_Return,
    mmm_lagged_return = Lagged_MMM_Return,
    pg_lagged_return = Lagged_PG_Return,
    gs_close = Close.y,
    gs_return = Return.y,
    mcd_close = Close.x.x,
    mcd_return = Return.x.x,
    mmm_close = Close.y.y,
    mmm_return = Returns,
    pg_close = Close,
    pg_return = Return.y.y
  )
# Dataset merge
data_combined <- reduce(list(jnj_returns, gs_returns_lagged, mcd_returns_lagged, mmm_returns_lagged, pg_returns_lagged), full_join, by = "Exchange Date")
# Final Dataset after removing NAs
final_data <- na.omit(data_combined)
# Merged datasets based on the Exchange date .
data_combined <- reduce(list(jnj_returns, gs_returns_lagged, mcd_returns, mmm_returns, pg_returns), full_join, by = "Exchange Date")
# Final Dataset
final_data <- na.omit(data_combined)

#########################################################################################################
# Classification-Based ML Methods

# Converting JNJ returns to binary format
final_data$JNJ_Return_Binary <- ifelse(final_data$Return.x > 0, 1, 0)

# training and testing sets
set.seed(123) 
index <- createDataPartition(final_data$JNJ_Return_Binary, p = 0.7, list = FALSE)

train_data <- final_data[index, ]
test_data <- final_data[-index, ]

train_data$gs_lagged_return<- lag(train_data$gs_lagged_return, 1)
train_data$gs_returns_lagged <- as.numeric(unlist(train_data$gs_lagged_return))
train_data$mcd_returns_lagged <- as.numeric(unlist(train_data$mcd_lagged_return))
train_data$mmm_returns_lagged <- as.numeric(unlist(train_data$mmm_lagged_return))
train_data$pg_returns_lagged <- as.numeric(unlist(train_data$pg_lagged_return))

log_model <- glm(JNJ_Return_Binary~ gs_lagged_return + mcd_lagged_return + mmm_lagged_return + pg_lagged_return, 
                 family = binomial(link = "logit"), data = train_data)


# Logistic Regression
log_model <- glm(JNJ_Return_Binary ~ gs_lagged_return + mcd_lagged_return + mmm_lagged_return + pg_lagged_return, 
                 family = binomial(link = "logit"), data = train_data)

#LDA
lda_model <- lda(JNJ_Return_Binary ~ gs_lagged_return + mcd_lagged_return + mmm_lagged_return + pg_lagged_return, 
                 +                  data = train_data)

# QDA
qda_model <- qda(JNJ_Return_Binary ~ gs_lagged_return + mcd_lagged_return + mmm_lagged_return + pg_lagged_return, 
                 data = train_data)

# Predictions on test set
log_pred <- predict(log_model, newdata = test_data, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

# Confusion Matrix

log_pred_class_factor <- factor(log_pred_class, levels = c("0", "1"))
test_data$JNJ_Return_Binary_factor <- factor(test_data$JNJ_Return_Binary, levels = c("0", "1"))
conf_mat <- confusionMatrix(log_pred_class_factor, test_data$JNJ_Return_Binary_factor)
print(conf_mat)
confusionMatrix(factor(log_pred_class), factor(test_data$JNJ_Return_Binary))


# ROC Curve and AUC
log_pred_roc <- prediction(log_pred, test_data$JNJ_Return_Binary)
log_perf <- performance(log_pred_roc, "tpr", "fpr")
plot(log_perf, colorize = TRUE)
auc(log_pred_roc)

# Assuming log_pred contains the predicted probabilities and test_data$JNJ_Return_Binary contains the actual binary outcomes

actual_outcomes <- as.numeric(as.factor(test_data$JNJ_Return_Binary)) - 1

# Prediction object for ROC
log_pred_roc <- prediction(log_pred, actual_outcomes)

#AUC CALCULATION
# AUC
print(auc_value)

# Assuming lda_model is your trained LDA model

# Predicting class probabilities
lda_pred_prob <- predict(lda_model, newdata = test_data)$posterior[,2]
lda_pred_class <- ifelse(lda_pred_prob > 0.5, 1, 0)

# Confusion Matrix for LDA
confusionMatrix(factor(lda_pred_class), factor(test_data$JNJ_Return_Binary))

# ROC Curve and AUC for LDA
lda_roc_curve <- roc(response = test_data$JNJ_Return_Binary, predictor = lda_pred_prob)
plot(lda_roc_curve)
lda_auc <- auc(lda_roc_curve)
print(lda_auc)

################################################################################
#Logistic reg
#test set 
log_pred_test_prob <- predict(log_model, newdata = test_data, type = "response")
# training set:
log_pred_train_prob <- predict(log_model, newdata = train_data, type = "response")



#LDA
# Test set
lda_pred_test_prob <- predict(lda_model, newdata = test_data)$posterior[,2]
# Training set:
lda_pred_train_prob <- predict(lda_model, newdata = train_data)$posterior[,2]

#QDA
# Test set
qda_pred_test_prob <- predict(qda_model, newdata = test_data)$posterior[,2]
# Training set:
qda_pred_train_prob <- predict(qda_model, newdata = train_data)$posterior[,2]


# Plotting ROC Curves and Calculating AUC for Logistic Regression
plot_roc_curve <- function(actual, predicted_prob, title = "ROC Curve") {
  # Ensure actual responses are in the correct format
  actual <- factor(actual, levels = c("0", "1"))
  
  # Generate the ROC object
  roc_obj <- roc(response = actual, predictor = as.numeric(predicted_prob))
  
  # Plot ROC curve
  plot(roc_obj, main = title)
  
  # Calculate AUC and print it
  auc_value <- auc(roc_obj)
  cat("AUC for", title, ":", auc_value, "\n")
}

# Assuming log_pred_train_prob contains the predicted probabilities for the training set from Logistic Regression
plot_roc_curve(train_data$JNJ_Return_Binary, log_pred_train_prob, "Logistic Regression - Train Set")
plot_roc_curve(test_data$JNJ_Return_Binary, log_pred_test_prob, "Logistic Regression - Test Set")

# Plotting ROC Curves and Calculating AUC for LDA
plot_roc_curve(train_data$JNJ_Return_Binary, lda_pred_train_prob, "LDA - Train Set")
plot_roc_curve(test_data$JNJ_Return_Binary, lda_pred_test_prob, "LDA - Test Set")

# Plotting ROC Curves and Calculating AUC for QDA
plot_roc_curve(train_data$JNJ_Return_Binary, qda_pred_train_prob, "QDA - Train Set")
plot_roc_curve(test_data$JNJ_Return_Binary, qda_pred_test_prob, "QDA - Test Set")

# Predicting class probabilities
qda_pred_prob <- predict(qda_model, newdata = test_data)$posterior[,2]
qda_pred_class <- ifelse(qda_pred_prob > 0.5, 1, 0)

# Confusion Matrix for QDA
confusionMatrix(factor(qda_pred_class), factor(test_data$JNJ_Return_Binary))

# ROC Curve and AUC for QDA
qda_roc_curve <- roc(response = test_data$JNJ_Return_Binary, predictor = qda_pred_prob)
plot(qda_roc_curve)
qda_auc <- auc(qda_roc_curve)
print(qda_auc)

################################################################################

x_train <- model.matrix(JNJ_Return_Binary ~  + mcd_lagged_return + mmm_lagged_return + pg_lagged_return, train_data)[,-1]
y_train <- train_data$JNJ_Return_Binary

# LASSO
lasso_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1)
# Ridge
ridge_model <- glmnet(x_train, y_train, family = "binomial", alpha = 0)
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0)

# Predictions and evaluation for LASSO 
lasso_pred_prob <- predict(cv_lasso, newx = model.matrix(~gs_lagged_return + mmm_lagged_return + pg_lagged_return, test_data)[,-1], s = "lambda.min", type = "response")
lasso_pred_class <- ifelse(lasso_pred_prob > 0.5, 1, 0)
# Confusion Matrix and ROC Curve for LASSO
confusionMatrix(factor(lasso_pred_class), factor(test_data$JNJ_Return_Binary))
roc_curve_lasso <- roc(response = test_data$JNJ_Return_Binary, predictor = as.numeric(lasso_pred_prob))
plot(roc_curve_lasso)
auc(roc_curve_lasso)
# Assuming `x_train` is your feature matrix and `y_train` are your outcomes:
x_train <- model.matrix(~ ., data = train_data)[, -1]  # exclude the intercept
y_train <- train_data$JNJ_Return_Binary
# cross-validation for LASSO
cv_fit_lasso <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")
#optimal lambda
plot(cv_fit_lasso)
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = cv_fit_lasso$lambda.min, family = "binomial")
# Predict using the lasso model with the new lambda
lasso_pred_prob <- predict(lasso_model, newx = model.matrix(~ ., data = test_data)[, -1], type = "response")
lasso_pred_class <- ifelse(lasso_pred_prob > 0.5, "1", "0")
test_data$JNJ_Return_Binary <- factor(test_data$JNJ_Return_Binary, levels = c("0", "1"))
lasso_pred_class <- factor(lasso_pred_class, levels = c("0", "1"))
test_data$JNJ_Return_Binary <- factor(test_data$JNJ_Return_Binary, levels = c("0", "1"))
# confusion matrix LASSO
confusion_matrix_lasso <- confusionMatrix(lasso_pred_class, test_data$JNJ_Return_Binary)
print(confusion_matrix_lasso)
#####################################################################################################
#Predictions and evaluation for Ridge

levels(test_data$JNJ_Return_Binary) <- c("0", "1")
levels(ridge_pred_class) <- c("0", "1")

#confusion matrix
confusion_matrix_ridge <- confusionMatrix(factor(ridge_pred_class, levels = c("0", "1")), 
                                          factor(test_data$JNJ_Return_Binary, levels = c("0", "1")))

ridge_pred_prob <- predict(cv_ridge, newx = model.matrix(~gs_lagged_return + mmm_lagged_return + pg_lagged_return, test_data)[,-1], s = "lambda.min", type = "response")

# Convert probabilities to class labels based on a threshold of 0.5
ridge_pred_class <- ifelse(ridge_pred_prob > 0.5, 1, 0)

# Evaluation for Ridge
# Confusion Matrix
confusion_matrix_ridge <- confusionMatrix(factor(ridge_pred_class), factor(test_data$JNJ_Return_Binary))

# ROC Curve and AUC for Ridge
roc_curve_ridge <- roc(response = test_data$JNJ_Return_Binary, predictor = as.numeric(ridge_pred_prob))
auc_ridge <- auc(roc_curve_ridge)

print(confusion_matrix_ridge)
plot(roc_curve_ridge)
print(auc_ridge)
##############################################################################################################
#########################################################################################################
#END





