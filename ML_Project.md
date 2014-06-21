# Predicting how well weight lifting exercise is performed


## Executive Summary
This report is aimed to predict the manner in which people did the weight lifting exercise based on provided data set. We use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. In this report, we will build a prediction model using random forest, run cross validation, provide the expected out of sample error, and then use the final model to predict a test data set having 20 test cases. The prediction correction rate on the test set is 100%. The data set used comes from this source: http://groupware.les.inf.puc-rio.br/har.


## Data Processing

### 1. Load data
First load the train and test data set and explore data.


```r
train <- read.csv("pml-training.csv", header = TRUE, na.strings = "NA")
test <- read.csv("pml-testing.csv", header = TRUE, na.strings = "NA")
dim(train)
```

```
## [1] 19622   160
```

```r
dim(test)
```

```
## [1]  20 160
```


### 2. Pre-processing data

Remove columns populated mostly with "", NULL and columns that are not good predictors such as X, user_name, timestamp and new_window.


```r
colRemoveList <- grep("^X|user_name|timestamp|new_window|kurtosis|skewness|max|min|amplitude|var|avg|stddev", 
    colnames(train))
train <- train[, -colRemoveList]
test <- test[, -colRemoveList]
dim(train)
```

```
## [1] 19622    54
```

```r
dim(test)
```

```
## [1] 20 54
```


### 3. Split train data sets for training and validation

```r
library(AppliedPredictiveModeling)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
set.seed(998)
trainIndex = createDataPartition(train$classe, p = 0.7, list = FALSE)
training = train[trainIndex, ]
validating = train[-trainIndex, ]
```


### 4. Train with Random Forest

```r
Sys.time()
```

```
## [1] "2014-06-21 14:12:48 EDT"
```

```r
trCtrl = trainControl(method = "cv", number = 4)
rf <- train(classe ~ ., data = training, method = "rf", trControl = trCtrl)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
Sys.time()
```

```
## [1] "2014-06-21 14:20:49 EDT"
```

```r
rf
```

```
## Random Forest 
## 
## 13737 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 10304, 10302, 10303, 10302 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##   2     1         1      7e-04        9e-04   
##   30    1         1      8e-04        0.001   
##   50    1         1      0.004        0.005   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```



### 5. Out of sample error for validating data set



```r
confusionMatrix(validating$classe, predict(rf, validating))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    1
##          B    0 1139    0    0    0
##          C    0    5 1021    0    0
##          D    0    0    4  960    0
##          E    0    0    0    4 1078
## 
## Overall Statistics
##                                         
##                Accuracy : 0.998         
##                  95% CI : (0.996, 0.999)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.997         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.996    0.996    0.996    0.999
## Specificity             1.000    1.000    0.999    0.999    0.999
## Pos Pred Value          0.999    1.000    0.995    0.996    0.996
## Neg Pred Value          1.000    0.999    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.183
## Detection Rate          0.284    0.194    0.173    0.163    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    0.998    0.998    0.998    0.999
```

```r
predictedValue <- predict(rf, validating)
1 - (sum(predictedValue == validating$classe)/nrow(validating))
```

```
## [1] 0.002379
```

The result shows the expected out of sample error is very low, less than 1%.

### 6. Predict the test data

```r
answers <- predict(rf, newdata = test)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


### 7. Generate files for submission

```r
pml_write_files <- function(x) {
    n <- length(x)
    for (i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, 
            col.names = FALSE)
    }
}
pml_write_files(answers)
```


## Result

After submitting the generated files to the course site, it shows all 20 test cases are correctly predicted. 
