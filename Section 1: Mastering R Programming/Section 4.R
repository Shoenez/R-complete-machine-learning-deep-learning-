library(caret)

set.seed(100)
data("segmentationData")
train_rows <- createDataPartition(segmentationData$Class, p=0.7, list=F)
trainData <- segmentationData[train_rows, -c(1:2)]
testData <- segmentationData[-train_rows, -c(1:2)]
table(segmentationData$Class)

set.seed(100)
ctrl <- trainControl(method='repeatedcv',
                     repeats=5,
                     summaryFunction = twoClassSummary, # to get AUROC
                     classProbs = TRUE,
                     sampling = 'down')

grid <- expand.grid(C = c(0.25, 0.5, 0.75, 1, 1,25, 1.5))

svmLin_mod <- train(Class~., data=trainData,
                    method = 'svmLinear',
                    preProc = c('center', 'scale'),
                    metric = 'ROC',
                    tuneGrid = grid,
                    trControl = ctrl)

svmLin_mod

# 17. Bagging and building Random Fs

set.seed(100)
data("segmentationData")
train_rows <- createDataPartition(segmentationData$Class, p=0.7, list=F)
trainData <- segmentationData[train_rows, -c(1:2)]
testData <- segmentationData[-train_rows, -c(1:2)]

ctrl <- trainControl(method='repeatedcv',
                     repeats=5,
                     summaryFunction = multiClassSummary, # to get AUROC
                     classProbs = TRUE)

grid <- expand.grid(mtry = c(2, 8, 15, 20, 30))
system.time({
  parRf_mod <- train(Class~., data = trainData, 
                     method = 'parRF',
                     preProc = c('center', 'scale'),
                     metric = 'ROC',
                     tuneGrid = grid,
                     trControl = ctrl)
})

pred <- predict(parRf_mod, testData)
caret::confusionMatrix(pred, testData$Class)