# prelimanaries ---------------------------------------------------------------
setwd("~/OneDrive/Coursera/Practical Machine Learning")
url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url_train, "./pml-training.csv", method="curl")
url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url_test, "./pml-testing.csv", method="curl")

training <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
testing <- read.csv("pml-testing.csv", na.strings = c("NA", "", "#DIV/0!"))
str(training) # appears some columns are empty (and thus have no predictive value)
colnames(training) == colnames(testing) # last column differs (classe vs problem_id)
is.factor(training$classe) # our response is correctly stored as a factor

# data cleaning and processing ------------------------------------------------
## removing features with little predictive value
### returns counts of NAs for each column
sum(apply(training, 2, function(x) length(which(is.na(x)))) == nrow(training)) # 6 columns completely empty
### using caret, find near zero variance predictors
library(caret)
nzv <- nearZeroVar(training, saveMetrics=T) # 36 features near zero variance
training <- training[, !nzv$nzv] # 124 features remain
testing <- testing[, !nzv$nzv]

## a more aggressive approach is to remove any columns with any NA values
### will follow this approach since it halves the number of features
### and aids in computational efficiency (necessary on my old machine)
training <- training[, colSums(is.na(training)) == 0] # 59 features remain
training <- training[, -c(1:6)] # remove first six
testing <- testing[, colSums(is.na(testing)) == 0]
testing <- testing[, -c(1:6)] # remove first six
colnames(training) == colnames(testing) # last column differs (classe vs problem_id)
# (Since data transformations may be less important in non-linear models like 
# classification trees, we do not transform any variables)
# creating (sub)training and validation set -----------------------------------
library(caret)
set.seed(1234)
inTrain <- createDataPartition(training$classe, p=0.7, list=F)
train <- training[inTrain, ]; dim(train)
valid <- training[-inTrain, ]; dim(valid)

rm(inTrain, nzv, training)

# classification tree ---------------------------------------------------------
library(tree)
## with default settings
treeDefault <- tree(classe ~ ., data=train, split="deviance")
summary(treeDefault) # 16 terminal nodes
plot(treeDefault)
text(treeDefault, cex=0.7, pretty=0)

# (1) GROW A FULL TREE:
## grow a bigger tree and prune back with cross-validation
stopCriteria <- tree.control(nobs=nrow(train), mincut=5, minsize=10, mindev=0.00001)
bigTree <- tree(classe ~ ., data=train, control=stopCriteria, split="deviance")
summary(bigTree) # big tree with many (unnecessary) terminal nodes
plot(bigTree)
# text(bigTree, cex=0.7, pretty=0)

# (2) COST COMPLEXITY PRUNING:
## bigTree overfits way too much (and perhaps we can predict better than treeDefault)
## use cross validation to prune back bigTree
crossval <- cv.tree(bigTree, K=10) # 10-fold cross validation on bigTree
crossval # size=num terminal nodes; dev=RSS; k=alpha (tuning parameter determining tree size)

plot(crossval$size, crossval$dev/nrow(train), type="b", xlab="Number of terminal nodes", 
     ylab="CV error", xlim=c(0, 60), col=ifelse(crossval$size == 16, "red", "black"),
     pch=ifelse(crossval$size==21, 19, 21), ylim=c(1.7, 3.2))
axis(3, at=crossval$size, lab=round(crossval$k)) # add alpha values to the plot
title("10-fold cross validation for classification tree", line=3.2)
## notice that 16 (or 21) terminal nodes are 'best' (16 was in fact the default tree)
## since minimal to no reduction in CV error beyond this point

# (3) DO THE PRUNING:
## will take 22 nodes to be 'best' for sake of illustrating pruning
treePruned <- prune.tree(bigTree, best=22)
plot(treePruned); text(treePruned, cex=0.7, pretty=0)

# predicting on validation set
pred_tree <- predict(treePruned, valid, type="class")
confusionTree <- confusionMatrix(valid$classe, pred_tree); confusionTree # 0.6753 accuracy (0.6472 on treeDefault)

## 0.6753 not enough accuracy to use for the quiz

rm(crossval, treeDefault, bigTree, stopCriteria)

# random forest ---------------------------------------------------------------
library(randomForest)
set.seed(1234)
rf <- randomForest(classe ~ ., data=train, ntree=200, importance=TRUE, na.action=na.exclude, do.trace=10)
rf # used default number of variables to consider at each split (mtry)

# choose number of trees:
head(rf$err.rate[, 1], 10)
plot(rf$err.rate[, 1], type="l", xlab="Number of trees", ylab="OOB error") # 100 trees is enough

# variable importance plot:
varImpPlot(rf, type=2)
importance(rf)
# type = 2 show reduction in gini index (class trees) and RSS (reg trees)
# see ?importance for "type" argument

# predicting on validation set
pred_rf <- predict(rf, valid)
confusionForest <- confusionMatrix(valid$classe, pred_rf); confusionForest # 0.9964 accuracy

## 0.9964 accuracy: feeling confident for the quiz

# use rf to predict on test set for quiz
pred_rf_quiz <- predict(rf, testing)
pred_rf_quiz # B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B (100%)