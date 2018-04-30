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
'%ni%' <- createDataPartition(bc$Class, p=0.7, list = F) #define not in func

options(scipen=999) #prevent scientific notaition


set.seed(100)


## Downsampling
## Upsampling
