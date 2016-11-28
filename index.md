# Practical Machine Learning
Matteo Tommasini  
November 28, 2016  

We refer to the instructions at the following
[link](https://www.coursera.org/learn/practical-machine-learning/peer/R43St/prediction-assignment-writeup) for an overview of the project. The dataset used in this report was provided in the following paper (see also [http://groupware.les.inf.puc-rio.br](http://groupware.les.inf.puc-rio.br) ):

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

## Loading the data


```r
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainingUrl, destfile="pml-training.csv",method="curl")
download.file(testingUrl, destfile="pml-testing.csv",method="curl")
complete.data <- read.csv("pml-training.csv"); partial.data <- read.csv("pml-testing.csv")
library(ggplot2); library(caret); library(rpart)
library(rattle); library(randomForest)
```

The 2 first 159 columns of the 2 data.frames `complete.data` and `partial.data` are equal. 


```r
all.equal(names(complete.data[,-160]),names(partial.data[,-160]))
```

```
## [1] TRUE
```

The last column (160-th column):

- (in the case of `complete.data`) is given by a column `classe` containing an integer between 1 and 5, associated to a class of exercise between "A" and "E"
- (in the case of `partial.data`) is given by a column `problem_id` simply repeating the names of the rows (i.e. the integers from 1 to 20), so it does not provide any useful info.

## Preliminary analysis and preprocess of the data

Looking at the data, it seems that the first column (called `X`), is simply a (progressive) number for the various measurements that were taken. So it should not be considered in our analysis.

In a real situation (when our analysis is supposed to be used in order to provide suggestions to a **new** user), we certainly want to extract general characteristics from the data, not depending on the 6 user that perfomed the activity during the test, therefore we should remove the column `user_name` from our analysis.

However, since the final aim of the present project is to predict the class of the activity on a set with the same 6 users (and since there is enough data for each user in `complete.data`), then we can take advantage of knowing the identity of the user. This is shown below:


```r
list.users <- unique(complete.data$user_name); print(as.character(sort(list.users)))
```

```
## [1] "adelmo"   "carlitos" "charles"  "eurico"   "jeremy"   "pedro"
```

```r
print(as.character(sort(unique(partial.data$user_name))))
```

```
## [1] "adelmo"   "carlitos" "charles"  "eurico"   "jeremy"   "pedro"
```

```r
sapply(seq_along(list.users), function(i) sum(complete.data$user_name == list.users[i]))
```

```
## [1] 3112 2610 3892 3536 3070 3402
```

Since there are only (exactly the same) 6 users both in `complete.data` and in `partial.data`, we can take full advantage of this fact by dividing our sets into 6 subsets, and using machine learning on each of these subset individually.


```r
list.complete <- list()
for(i in seq_along(list.users))
  list.complete[[i]] <- complete.data[complete.data$user_name == list.users[i],]
```

Moreover, we want to split `complete.data` into training/validating/test sets with percentages 60/20/20. So we proceed as follows:


```r
set.seed(12345)
list.training <- list(); list.validating <- list(); list.testing <- list()
for(i in seq_along(list.users)){
  # % 60% of the data are used for the training set
  inTrain <- createDataPartition(y=list.complete[[i]]$classe, p = 0.60)[[1]]
  list.training[[i]] <- list.complete[[i]][inTrain,]
  # half of the remaining data (i.e. 20% of the total) are used for validating
  temp <- list.complete[[i]][-inTrain,]
  inValidate <- createDataPartition(y=temp$classe, p = 0.50)[[1]]
  list.validating[[i]] <- temp[inValidate,]
  # The remaining 20% is used for testing the model
  list.testing[[i]] <- temp[-inValidate,]
}
```

As a check, we print the dimensions of the various training and validating sets:


```r
for(i in seq_along(list.users)){
  print(paste(list.users[i], " = ", dim(list.training[[i]])[1], ", ",
              dim(list.validating[[i]])[1], ", ", dim(list.testing[[i]])[1], sep = ""))
}
```

```
## [1] "carlitos = 1869, 623, 620"
## [1] "pedro = 1568, 522, 520"
## [1] "adelmo = 2336, 778, 778"
## [1] "charles = 2124, 707, 705"
## [1] "eurico = 1845, 613, 612"
## [1] "jeremy = 2045, 679, 678"
```

A preliminary analysis shows that each training and validating set has lots of NA values.
We opt for removing all the columns where more than half of the column is composed of NAs.


```r
# This chunk of code takes around 2 minutes to execute on a laptop.
# In order to use the same columns both in training/validating/testing, we have to keep
# track of the variables that we select (for each user separately). These are saved in the
# following list (contaning 6 vector of variables):
list.variables <- list();
for(i in seq_along(list.users)){
  # Counts how many empty cells or "#DIV/0!" cells or "NA" cells there are in each column.
  d1 <- dim(list.training[[i]])[1]; d2 <- dim(list.training[[i]])[2]
  # This list below will contain the number of cells per column that have problems (NA, etc.).
  # The first column contains identifiers for the the various observations,
  # and using it will lead to false predictions. So the first counter is set to d1.
  count <- rep(x=0,d2);   count[1] <- d1;
  for(k in 2 : d2){
    for(j in 1 : d1)
      if(is.na(list.training[[i]][j,k]) |
           list.training[[i]][j,k] %in% c("#DIV/0!",""))
          count[k] <- count[k] + 1
  }
  # We select only the columns with less than half of the values missing.
  good.variables <- count < (d1 * 0.5)
  # Actually a direct check with th command
  #    print(table(count[good.variables]))
  # proves that the variables that we selected have no missing values, 
  # so there is no need for imputing.
  # We select only the variables of interest.
  list.training[[i]] <- list.training[[i]][,good.variables]
  # In addition, we need to check for almost zero variability for the remaining variables
  # By construction, the column "user_name" is constant (on each subset with constant "i"),
  # but there could also be other constant or almost constant columns to remove.
  # The next line proves that actually only such column is constant.
  near.zero.var <- nearZeroVar(list.training[[i]], saveMetrics = TRUE)$nzv
  # because the only TRUE value is the first one
  list.training[[i]] <- list.training[[i]][,-near.zero.var]

  # We save the list of the variables used for the i-th user
  list.variables[[i]] <- colnames(list.training[[i]])

  # For later use, we select the same variable also in validating and in testing.
  list.validating[[i]] <- list.validating[[i]][,list.variables[[i]]]
  list.testing[[i]] <- list.testing[[i]][,list.variables[[i]]]
}
```

A simple check proves that actually we select the same set of (58) variables for the 6 users
(something that a priori we cannot expecting), but we are not going to use this fact



## Creation of 3 different models using the training set (for each user separately)

Given the high heterogeneity of the types of the variables in `complete.data`, we opt for using random forest models. However, since each trainining set is still too big, we decide to create 3 subsets out of each training set (for each user), fit a random forest model for each of these subsets, then validate their combination using the validating part. So the final division of the data for each user will be:

- 20% used for training1,
- 20% used for training2,
- 20% used for training3,
- 20% used for validating the combination of the previous 3 model,
- 20% for the testing of the final model.

So divide our training sets in 3 subsets, and we fit a random forest model to each of them.


```r
list.fit1 <- list();   list.fit2 <- list();   list.fit3 <- list();
for(i in seq_along(list.users)){
  train1 <- createDataPartition(y = list.training[[i]]$classe, p = 1/3, list = FALSE)
  subsets1 <- list.training[[i]][train1,]
  temp <- list.training[[i]][-train1,]
  train2 <- createDataPartition(y = temp$classe, p = 1/2, list = FALSE)
  subsets2 <- temp[train2,]
  subsets3 <- temp[-train2,]
  list.fit1[[i]] <- train(classe ~ ., method = "rf", data = subsets1,
                                 trControl = trainControl(method = "cv"), number = 3)
  list.fit2[[i]] <- train(classe ~ ., method = "rf", data = subsets2,
                                 trControl = trainControl(method = "cv"), number = 3)
  list.fit3[[i]] <- train(classe ~ ., method = "rf", data = subsets3,
                                 trControl = trainControl(method = "cv"), number = 3)
}
```

## Validating and final model

For each user separately, we use the 3 fit models already created and we combine them, again using random forests. This results in a `combo.model` for each user. At the end, we compute the accuracy for each user individually.


```r
combo.model <- list();    accuracy <- rep(NA,length(list.users))
for(i in seq_along(list.users)){
  pred1<- predict(list.fit1[[i]], list.validating[[i]])
  pred2 <- predict(list.fit2[[i]], list.validating[[i]])
  pred3 <- predict(list.fit3[[i]], list.validating[[i]])
  validating.dat <- data.frame(pred1, pred2, pred3, classe = list.validating[[i]]$classe)
  combo.model[[i]] <- train(classe ~ ., method = "rf", data = validating.dat)
  
  pred.test1 <- predict(list.fit1[[i]], newdata = list.testing[[i]])
  pred.test2 <- predict(list.fit2[[i]], newdata = list.testing[[i]])
  pred.test3 <- predict(list.fit2[[i]], newdata = list.testing[[i]])
  
  testing.dat <- data.frame(pred1 = pred.test1, pred2 = pred.test2, pred3 = pred.test3,
                            classe = list.testing[[i]]$classe)
  pred <- predict(combo.model[[i]], newdata = testing.dat)
  accuracy[i] <- confusionMatrix(pred, list.testing[[i]]$classe)$overall[1]
}
print(t(data.frame(user = 1:length(list.users), accuracy)))
```

```
##               [,1] [,2] [,3]      [,4]     [,5] [,6]
## user     1.0000000    2    3 4.0000000 5.000000    6
## accuracy 0.9951613    1    1 0.9971631 0.996732    1
```

So the accuracy obtained with out-of-sample is between 0.995 and 1 (depending on the various users). If we consider the 20 observations of the set `partial.data`, then the predicted corrected number of values out of 20:


```r
t <- table(partial.data$user_name)
sum(t*accuracy)
```

```
## [1] 19.95767
```

So we feel confortable enough to predict the 20 values of `classe` out of the data in `partial.data`:


```r
d <- dim(partial.data)[1]
final.predictions <- rep(NA,d);   v <- c("A","B","C","D","E")
for( k in 1:d ){
  i <- which(list.users == partial.data$user_name[k])
  temp <- partial.data[k,]
  pred.final1 <- predict(list.fit1[[i]], newdata = temp)
  pred.final2 <- predict(list.fit2[[i]], newdata = temp)
  pred.final3 <- predict(list.fit2[[i]], newdata = temp)
  final.data <- data.frame(pred1 = pred.final1, pred2 = pred.final2, pred3 = pred.final3)
  final.predictions[k]  <- predict(combo.model[[i]], newdata = final.data)
}
final.predictions <- v[final.predictions];
print(t(data.frame(id = 1:d,prediction = final.predictions)))
```

```
##            [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
## id         " 1" " 2" " 3" " 4" " 5" " 6" " 7" " 8" " 9" "10"  "11"  "12" 
## prediction "B"  "A"  "B"  "A"  "A"  "E"  "D"  "B"  "A"  "A"   "B"   "C"  
##            [,13] [,14] [,15] [,16] [,17] [,18] [,19] [,20]
## id         "13"  "14"  "15"  "16"  "17"  "18"  "19"  "20" 
## prediction "B"   "A"   "E"   "E"   "A"   "B"   "B"   "B"
```

This prediction is the correct one: if we insert these results in the final [quiz](https://www.coursera.org/learn/practical-machine-learning/exam/3SSqy/course-project-prediction-quiz), we get 20 correct answers out of 20.
