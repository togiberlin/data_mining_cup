# Load the required packages
library(caret)
library(FSelector)
set.seed(42)

# Load the train and test datasets
train_df<-read.csv("training_dataDMC2.csv")
test_df<-read.csv("test_datsaDMC2.csv")

# Get the structure of the train dataset
str(train_df)

# Find the columns with the NA values

colSums(is.na(train_df)) ## we do not have missing values


#Convert Variables to factors

train_df$engine_type<-as.factor(train_df$engine_type)
train_df$vehicle_type<-as.factor(train_df$vehicle_type)

test_df$engine_type<-as.factor(test_df$engine_type)
test_df$vehicle_type<-as.factor(test_df$vehicle_type)


# Replace the levels of defect n to 0 y to 1
levels(train_df$defect)[which(levels(train_df$defect)=="y")]<-"1"
levels(train_df$defect)[which(levels(train_df$defect)=="n")]<-"0"


#Remove the irrelevant variables from the predictive model

train_df<-subset(train_df, select=-c(system_readout_id, vehicle_identification_number ))



####################################################################
## Feature Selection

# Calculate weights for the attributes using Info Gain and Gain Ratio



weights_info_gain<-information.gain(defect ~ ., data=train_df)
weights_info_gain

weights_gain_ratio = gain.ratio(defect ~ ., data=train_df)
weights_gain_ratio


# Select the 50 most important attributes based on Gain Ratio
most_important_attributes <- cutoff.k(weights_gain_ratio, 50)
most_important_attributes

formula_with_most_important_attributes <- as.simple.formula(most_important_attributes, "defect")
formula_with_most_important_attributes




####################################################################
# Training & Evaluation
# 3 x 5-fold cross validation
fitCtrl = trainControl(method="repeatedcv", number=5, repeats=3)


# training a Decision Tree, One Rule, Boosting Models, Naive Bayes, GLM, KNN using the metric "Accuracy"
modelDT = train(formula_with_most_important_attributes, data=train_df, method="J48", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelOneR = train(formula_with_most_important_attributes, data=train_df, method="OneR", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelBoost = train(formula_with_most_important_attributes, data=train_df, method="LogitBoost", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelnb = train(formula_with_most_important_attributes, data=train_df, method="nb", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelGLM = train(formula_with_most_important_attributes, data=train_df, method="glm", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)
modelKNN = train(formula_with_most_important_attributes, data=train_df, method="knn", trControl=fitCtrl, metric="Accuracy",na.action = na.pass)

# Compare results of different models
res = resamples(list(dt=modelDT,oneR=modelOneR, boost =modelBoost, nb=modelnb, GLM=modelGLM, KNN=modelKNN))
summary(res)


# Show confusion matrix (in percent)
confusionMatrix(modelDT)
confusionMatrix(modelBoost)
confusionMatrix(modelOneR)
confusionMatrix(modelnb)
confusionMatrix(modelGLM)
confusionMatrix(modelKNN)


# The best model appears to be the GLM with average accuracy 78.7%





####################################################################
## Predict classes in test data

prediction_classes = predict.train(object=modelGLM, newdata=test_df, na.action=na.pass)
predictions = data.frame(id=test_df$system_readout_id, prediction=prediction_classes)


######################################################
# Export the Predictions
write.csv(predictions, file="predictions_Unic0rn_2.csv", row.names=FALSE)
