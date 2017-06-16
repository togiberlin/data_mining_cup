# Load the required packages
library(caret)
library(FSelector)
set.seed(42)

# Load the train and test datasets
train_df<-read.csv("training_dataDMC1.csv")
test_df<-read.csv("test_dataDMC1.csv")

# Get the structure of the train dataset
str(train_df)


####################################################################
## Data Preparation


## Replace the NA values with the Mode

# Find the columns with the NA values

colSums(is.na(train_df))


train_df$Job[is.na(train_df$Job)]<-names(which.max(table(train_df$Job)))
train_df$Education[is.na(train_df$Education)]<-names(which.max(table(train_df$Education)))
train_df$Communication[is.na(train_df$Communication)]<-names(which.max(table(train_df$Communication)))
train_df$Outcome[is.na(train_df$Outcome)]<-names(which.max(table(train_df$Outcome)))

# The same for the test dataset using the mode of the train dataset
test_df$Job[is.na(test_df$Job)]<-names(which.max(table(train_df$Job)))
test_df$Education[is.na(test_df$Education)]<-names(which.max(table(train_df$Education)))
test_df$Communication[is.na(test_df$Communication)]<-names(which.max(table(train_df$Communication)))
test_df$Outcome[is.na(test_df$Outcome)]<-names(which.max(table(train_df$Outcome)))

######################################################
# Convert the CarInsurance to factor
train_df$CarInsurance<-as.factor(train_df$CarInsurance)


# Create a column called "Duration" which is the call duration

train_df$Duration<-as.numeric(as.POSIXct(train_df$CallEnd, format="%H:%M:%S")-as.POSIXct(train_df$CallStart, format="%H:%M:%S"))
test_df$Duration<-as.numeric(as.POSIXct(test_df$CallEnd, format="%H:%M:%S")-as.POSIXct(test_df$CallStart, format="%H:%M:%S"))

# Convert the columns Default, HHInsurance, CarLoan to factors

train_df$Default<-as.factor(train_df$Default)
train_df$HHInsurance<-as.factor(train_df$HHInsurance)
train_df$CarLoan<-as.factor(train_df$CarLoan)

test_df$Default<-as.factor(test_df$Default)
test_df$HHInsurance<-as.factor(test_df$HHInsurance)
test_df$CarLoan<-as.factor(test_df$CarLoan)


# Factorize the DaysPassed to Contact / no Contact

train_df$IsContact<-factor(rep("NA", nrow(train_df)), levels=c("0", "1"))
train_df$IsContact[train_df$DaysPassed>-1]<-"1"
train_df$IsContact[train_df$DaysPassed==-1]<-"0"

test_df$IsContact<-factor(rep("NA", nrow(test_df)), levels=c("0", "1"))
test_df$IsContact[test_df$DaysPassed>-1]<-"1"
test_df$IsContact[test_df$DaysPassed==-1]<-"0"



####################################################################
## Feature Selection

# Calculate weights for the attributes using Info Gain and Gain Ratio

# In the model we will exlude the following Columns:
# Id because it is the row ID
# CallStart and CallEnd because we have built the Column Duration which is the difference of those
# DaysPassed because based on this Column we created the IsContact which is a factor 

train_df<-subset(train_df, select=-c(Id, CallStart, CallEnd, DaysPassed))

weights_info_gain<-information.gain(CarInsurance ~ ., data=train_df)
weights_info_gain

weights_gain_ratio = gain.ratio(CarInsurance ~ ., data=train_df)
weights_gain_ratio


# Select the 12 most important attributes based on Gain Ratio
most_important_attributes <- cutoff.k(weights_gain_ratio, 12)
most_important_attributes

formula_with_most_important_attributes <- as.simple.formula(most_important_attributes, "CarInsurance")
formula_with_most_important_attributes


####################################################################
# Training & Evaluation
# 3 x 5-fold cross validation
fitCtrl = trainControl(method="repeatedcv", number=5, repeats=3)


# Training a Decision Tree, One Rule, Boosting Models, Naive Bayes, GLM, KNN using the metric "Accuracy"
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


# Based on the results the top models are the Decision Tree and GLM. But it seems slightly higher for the DT and for that reason this model will be chosen. 
# The accuracy of the model is 81.71% 
##########################



####################################################################
## Predict classes in test data

prediction_classes = predict.train(object=modelDT, newdata=test_df, na.action=na.pass)
predictions = data.frame(id=test_df$Id, prediction=prediction_classes)


######################################################
# Export the predictions
write.csv(predictions, file="predictions_Unic0rn_2.csv", row.names=FALSE)
