#12. Log regression ...

library(mlbench)
data("BreastCancer", package = "mlbench")
bc <- BreastCancer[complete.cases(BreastCancer),]
dim(bc)
str(bc)

bc <- bc[,-1]

## Convert to numeric 
for(i in 1:9) {
  bc[,i] <- as.numeric(as.character(bc[,i]))
}

bc$Class <-ifelse(bc$Class == "malignant", 1,0)
bc$Class <-factor(bc$Class, levels = c(0,1))
table(bc$Class)

## Train & Test
library(caret)
'%ni%' <-  Negate('%in%')#define not in func
options(scipen=999) #prevent scientific notaition

set.seed(100)
trainDataIndex <- createDataPartition(bc$Class, p=0.7, list = F)
trainData <- bc[trainDataIndex,]
testData <- bc[-trainDataIndex,]

## Downsampling
set.seed(100)
down_train <- downSample(x = trainData[,colnames(trainData) %ni% "Class"] , y = trainData$Class)
table(down_train$Class)
## Upsampling
up_train <- upSample(x = trainData[,colnames(trainData) %ni% "Class"] , y = trainData$Class)
table(up_train$Class)

## Hybrid

##Logit Model
logitmod <- glm(Class ~ Cl.thickness + Cell.size + Cell.shape, family = binomial, data = down_train)
summary(logitmod)

pred <- predict(logitmod, newdata = testData, type = 'response')

y_pred_num <- ifelse(pred > 0.5, 1, 0)
y_pred <- factor(y_pred_num, levels = c(0,1))
y_act <- testData$Class

y_pred
y_act
mean(y_pred == y_act)

caret::confusionMatrix(y_pred, y_act, positive = '1')

library(InformationValue)
InformationValue::plotROC(y_act, pred)
InformationValue::AUROC(y_act,pred)

# Challenge 12 

lgit1 <- glm(Class ~ Cl.thickness + Mitoses + Bl.cromatin, data = down_train, family = binomial)
summary(lgit1)
pred1 <- predict(lgit1, newdata = testData, type = 'response')

y_pred_num1 <- ifelse(pred1 > 0.5, 1, 0)
y_pred1 <- factor(y_pred_num1, levels = c(0,1))

caret::confusionMatrix(y_pred1, y_act, positive = '1')

InformationValue::plotROC(y_act, pred1)
InformationValue::AUROC(y_act,pred1)

lgit2 <- glm(Class ~ Cell.size + Marg.adhesion, data = down_train, family = binomial)
pred2 <- predict(lgit2, newdata = testData, type = 'response')
y_pred_num2 <- ifelse(pred2 > 0.5, 1, 0)
y_pred2 <- factor(y_pred_num2, levels = c(0,1))
caret::confusionMatrix(y_pred2, y_act, positive = '1')
InformationValue::plotROC(y_act, pred2)
InformationValue::AUROC(y_act,pred2)

# 13. Naïve Bayes

data(Vehicle, package = 'mlbench')
head(Vehicle)
str(Vehicle)

library(caret)
set.seed(100)
train_rows <- caret::createDataPartition(Vehicle$Class, p=0.7, list=F)
train <- Vehicle[train_rows,]
test <- Vehicle[-train_rows,]

caret::featurePlot(Vehicle[,-19], Vehicle[,19], plot ="box")

library(klaR)
nb_mod <- NaiveBayes(Class ~ ., data=train)
pred <- predict(nb_mod, test)
mean(test$Class != pred$class)

tab <- table(pred$class, test$Class)
caret::confusionMatrix(tab)
plot(nb_mod)

# 13. Challenge

data("iris", package = 'datasets')
set.seed(100)
train_rows <- caret::createDataPartition(iris$Species, p=0.7, list=F)
train <- iris[train_rows,]
test <- iris[-train_rows,]

nb_iris <- NaiveBayes(Species ~., data = train)
pred_i <- predict(nb_iris, test)
mean(test$Species != pred_i$class)

tab_i <- table(pred_i$class, test$Species)
caret::confusionMatrix(tab_i)

# 14. Trees

library(caret)
set.seed(100)
train_rows <- createDataPartition(iris$specis, p=0.7, list=F)
trainData <- iris[train_rows,]
testData <- iris[-train_rows,]

library(partykit)
ctMod <- ctree(Species ~., data = trainData)
print(ctMod)
plot(ctMod)

ctMod2 <- ctree(Species ~., data = trainData, control = ctree_control(maxdepth = 2))
print(ctMod2)
plot(ctMod2)

out <- predict(ctMod2, testData)
mean(test[,5] != out)
sum(test[,5] != out)

library(rpart)
rpartMod <- rpart(Species ~., data = trainData, control = rpart.control(minsplit=5, cp=0, maxdepth = 4))
pred <- predict(rpartMod, testData)
mean(pred != as.character(testData$Species))

iris_party <- as.party.rpart(rpartMod)
plot(iris_party)

# 16.

library(caret)

preProcessParams <- preProcess(iris[,1:4], method=c('range'))
preProcessParams

normalized <- predict(preProcessParams, iris[,1:4])
iris_n <- cbind(normalized, Species = iris$Species)

summary(iris_n)
 
# caret train package info here: topepo.github.io/caret/index.html

set.seed(100)
train_rows <- createDataPartition(iris_n$Species, p = 0.7, list = F)
train_data <- iris_n[train_rows,]
test_data <- iris_n[-train_rows,]
fit <- train(Species~., data=train_data, preProcess=c('range'), method = 'knn')

predict(fit$finalModel, newdata = test_data[, 1:4], type = 'class')

tc <- trainControl(method='repeatedcv', number = 5, repeats = 3)
tc

# Task

data(Vehicle, package = 'mlbench')
set.seed(100)
vpreProcessParams <- preProcess(Vehicle[,-19], method=c('range'))
vnormal <- predict(vpreProcessParams, Vehicle[,-19])
veh_n <- cbind(vnormal, Class = Vehicle$Class)

train_rows <- createDataPartition(veh_n$Class, p = 0.7, list = F)
vtrainData <- veh_n[train_rows,]
vtestData <- veh_n[-train_rows,]
library(doParallel)
registerDoParallel(4)
tc <- trainControl(method = "repeatedcv",
                   number = 5, repeats = 3,
                   search = 'random',
                   summaryFunction = multiClassSummary)

vfit <- train(Class~., data=vtrainData, method='C5.0', trainControl=tc,
             metric='Kappa', tunelength = 5)
mod <- fit$finalModel
pred <- predict(mod, vtestData)

testResults <- predict(mod, vtestData, type='prob') # calculate class probs
testResults <- data.frame(testResults)
testResults$obs <- vtestData$Class
testResults$pred <- predict(vfit, vtestData, type='raw')
multiClassSummary(testResults, lev - levels(vtestData$Class))

# 17. 

data('GlaucomaM', package = 'TH.data')

set.seed(100)
train_rows <- createDataPartition(GlaucomaM$Class, p=0.7, list = F)
trainData <- GlaucomaM[train_rows,]
testData <- GlaucomaM[-train_rows,]

library(Boruta)
borutamod <- Boruta(Class~., data = trainData, doTrace = 1)
borutamod

borutaSignif <- getSelectedAttributes(borutamod, withTentative = TRUE)
print(borutaSignif)

roughFixMod <- TentativeRoughFix(borutamod)
borutaSignif <- getSelectedAttributes(roughFixMod, withTentative = TRUE)
print(borutaSignif)

plot(roughFixMod, cex.axis=.7, las=2, xlab = "", main = 'Variable Importance')

## Variable importance

### RPart
rpartMod <- train(Class~., data = trainData, method='rpart')
rpartvar <- varImp(rpartMod)
rpartvar

### Random F
rforMod <- train(Class~., data = trainData, method='rf')
rforvar <- varImp(rforMod)
rforvar

## Recrusive Feature Elimination

x <- trainData[,-63]
y <- trainData[,63]
test_x <- testData[,-63]
## Number of fetures to be retained
subsets <- c(1:5, 10, 15, 20, 25, 35, 45, 55)

## Removes Highly correlated variables
correls = findCorrelation(cor(x), cutoff = .9)
if (length(correls) != 0){
  x<-x[,-correls]
}
set.seed(100)
index <- createFolds(y, k=10, returnTrain = T)

control <- rfeControl(functions = rfFuncs, 
                      method='repeatedcv',
                      repeats = 5,
                      index=index,
                      verbose=T)
rfProfile <- rfe(x=x, y=y, 
                 sizes=subsets,
                 rfeControl = control)
rfProfile
varImp(rfProfile)
names(rfProfile)
rfProfile$optVariables

# Task
set.seed(100)
data("segmentationData")
train_rows <- createDataPartition(segmentationData$Class, p=0.7, list=F)
sDtrainData <- segmentationData[train_rows,]
sDtestData <- segmentationData[-train_rows,]

## Boruta
Bmod <- Boruta(Class~., data=sDtrainData, doTrace=2)
Bsignif <- getSelectedAttributes(Bmod, withTentative = T)
print(Bsignif)

## RFE
sdx <- sDtrainData[,-c(1,2,3)]
sdy <- sDtrainData[,3]
test_x <- sDtestData[,-c(1,2,3)]

set.seed(100)
index <- createFolds(y, k=10, returnTrain = T)

control <- rfeControl(functions = rfFuncs, 
                      method='repeatedcv',
                      repeats = 5,
                      index=index,
                      verbose=T)
rfProfile <- rfe(x=x, y=y, 
                 sizes=subsets,
                 rfeControl = control)
rfProfile
varImp(rfProfile)
names(rfProfile)
rfProfile$optVariables
