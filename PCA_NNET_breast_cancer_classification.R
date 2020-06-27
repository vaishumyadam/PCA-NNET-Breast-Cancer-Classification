## Vaishu Myadam (vmyadam1208@gmail.com)
## June, 2020

## Using the Wisconsin Breast Cancer dataset

# Necessary libraries

library("e1071")
library(caret)
library(pROC)
library(gridExtra)
library(grid)
library(ggfortify)
library(purrr)
library(dplyr)
library(reshape2)
library(readr)
library(corrplot)
require(foreach)
require(iterators)
require(parallel)
library(nnet)
library(doParallel)
registerDoParallel()

# Reading in data

wisconsindata = read.csv("wisconsindata.csv", sep = ",")

# Data cleaning

str(wisconsindata) # Seeing a summary to decide what features to remove

cleaned_data = wisconsindata[,-c(0:1)] # Removing unnecessary ID column
cleaned_data = cleaned_data[, -32] # Removing useless last column
cleaned_data$diagnosis = as.factor(cleaned_data$diagnosis) # Tidying the dataset

summary(cleaned_data)

# Removing unnecessary predictors (bivariate multivariate analysis)

correlations = cor(cleaned_data[,2:31])
corrplot(correlations, order = "original", tl.cex = 0.5)
highly_correlated_features = colnames(cleaned_data)[findCorrelation(correlations, cutoff = 0.9, verbose = TRUE)]
cleaned_data_cor = cleaned_data[, which(!colnames(cleaned_data) %in% highly_correlated_features)]

# Visualization

diagnosis_list = table(cleaned_data$diagnosis)
diagnosis_proportions = prop.table(diagnosis_list) * 100
pielabels = sprintf("%s - %3.1f%s", c("Benign", "Malignant"), diagnosis_proportions, "%")

pie(diagnosis_proportions,
    labels = pielabels,  
    clockwise = TRUE,
    col= c("green", "red"),
    border="black",
    radius = 0.8,
    cex = 1, 
    main="Cancer Diagnosis")
legend(1, .5, legend = c("Benign", "Malignant"), cex = 1, fill = c("green", "red"))

# Data preprocessing (principal component analysis)

pca_data = prcomp(cleaned_data[, 2:31], center=TRUE, scale=TRUE)
plot(pca_data, type="l", main='Principal Components Weight')
grid(nx = 10, ny = 10)
title(main = "Principal Components Weight", sub = NULL, xlab = "Principal Components")
box()

summary(pca_data) # To see the difference from non pca data

# Removing highly correlated features again
pca_data_cleaned = prcomp(cleaned_data_cor, center=TRUE, scale=TRUE)
summary(pca_data_cleaned)
pca_df = as.data.frame(pca_data_cleaned$x)

# Splitting training and testing data
set.seed(1208)
complete_dataset = cbind(diagnosis = cleaned_data$diagnosis, cleaned_data_cor)
index = createDataPartition(complete_dataset$diagnosis, p = 0.7, list = FALSE)

training_set = complete_dataset[ index,]
testing_set = complete_dataset[-index,]

# Building the model

fitControl = trainControl(method="cv",
                           number = 5,
                           preProcOptions = list(thresh = 0.99), 
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary)

pca_nnet_model = train(diagnosis~.,
                        data = training_set,
                        method="nnet",
                        metric="ROC",
                        preProcess=c('center', 'scale', 'pca'),
                        tuneLength=10,
                        trace=FALSE,
                        trControl = fitControl)
# Presenting results

predicted_pca_nnet = predict(pca_nnet_model, testing_set)
confusion_matrix_pca_nnet = confusionMatrix(predicted_pca_nnet, testing_set$diagnosis, positive = "M")
confusion_matrix_pca_nnet

confusion_table <- as.table(confusion_matrix_pca_nnet, nrow = 2, byrow = TRUE)
fourfoldplot(confusion_table, color = c("red", "green"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")

