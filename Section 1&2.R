## Univariate Analysis
library(MASS)
install.packages("pastecs")
pastecs::stat.desc(Cars93)

quantile(Cars93$Price, 0.95)

## Bivariate Analysis

data(Prestige, package = "car")
plot(x=Prestige$education, y = Prestige$income,
     main = "Income vs Education",
     ylim = c(0,10000))
abline(lm(income ~ education, data = Prestige))

# c cor function
cor(Prestige$income, Prestige$education)

cor.test(Prestige$income, Prestige$education)

# Challenge

m <- lm(income ~., data = Prestige)
anova(m)

# 4. Detecting and treating Outlier

var <- Prestige$income

ninetyn_quartile <- quantile(var, 0.99)
first_quantile <- quantile(var, 0.01)

var[var > ninetyn_quartile] <- ninetyn_quartile
var[var < first_quantile] <- first_quantile

# 5. Treating Outliers with mice

Prestige_miss <- read.csv("https://raw.githubusercontent.com/selva86/datasets/master/Prestige_miss.csv")
myData <- Prestige_miss
head(myData)

library("Hmisc")
myData$education <- impute(myData$education, mean)
myData$type <- impute(myData$type, mode)

library("mice")
myData <- Prestige_miss
micemod <- mice(myData)

myData2 <- complete(micemod, 1)


#6. Linear Regression

data(Prestige, package = "car")
Prestige <- na.omit(Prestige)
set.seed(100)
train_rows <- sample(1:nrow(Prestige), size = 0.7*nrow(Prestige))
training <- Prestige[train_rows,]
test <- Prestige[-train_rows,]
lmmod <- lm(prestige ~ income + education, data = training)

predicted <- predict(lmmod, newdata = training)
mean((training$prestige - predicted)^2)
mean(abs((training$prestige - predicted)/training$prestige))

#9. Best subsets and step-wise regression challenge
library(car)
base.mod <- lm(prestige ~ 1, data = training)
all.mod <- lm(prestige ~., data = training)

stepMod <- step(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = 'both', trace = 1, steps = 1000)
stepMod
vif(stepMod)

#11. Non-linear regressors

plot(x=training$income, y = training$prestige)
sp_2 <- smooth.spline(x=training$income, y = training$prestige, df = 2)
sp_20 <- smooth.spline(x=training$income, y = training$prestige, df = 20)
sp_50 <- smooth.spline(x=training$income, y = training$prestige, df = 50)
sp_10 <- smooth.spline(x=training$income, y = training$prestige, df = 10)
sp_cv <- smooth.spline(x=training$income, y = training$prestige, cv = T)
sp_cv$df

plot(x=training$income, y = training$prestige, main = "Income vs Prestige")
lines(sp_2, lwd = 2, col = "blue")
lines(sp_10, lwd = 2, col = "lightblue")
lines(sp_20, lwd = 2, col = "green")
lines(sp_50, lwd = 2, col = "darkgreen")
lines(sp_cv, lwd = 3, col = "red")

predicted <- predict(sp_cv, test$income)$y
DMwR::regr.eval(test$prestige, predicted)

library(mgcv)
library(splines)

head(ns(Prestige$income, df=3))

gamMod <- mgcv::gam(prestige ~ ns(income, 3) + ns(education,4) + type, data = training)
predicted <- predict(gamMod, test)
DMwR::regr.eval(test$prestige, predicted)

# Challenge 

cars1 <- cars[1:30, ]  # original data
cars_outliers <- data.frame(speed=c(19,19,20,20,20), dist=c(190, 186, 210, 220, 218))  # introduce outliers.
cars2 <- rbind(cars1, cars_outliers)  # data with outliers.

# plot
plot(cars2$speed, cars2$dist, pch="*", col="red", main="Dist Vs. Speed", xlab="Speed", ylab="Dist")
abline(lm(dist ~ speed, data=cars2), col="blue", lwd=3, lty=2)

# Fit linear model and plot
lmmod <- lm(dist ~ speed, data=cars2)  # fit model
predicted_lm <- predict(lmmod, cars2)  # predict
DMwR::regr.eval(cars2$dist, predicted_lm)  # Errors

head(ns(cars2$speed, df = 3))
gam_car <- gam(dist ~ ns(speed, 3), data = cars2)
pred_car <- predict(gam_car, cars2)
DMwR::regr.eval(cars2$dist, pred_car)

abline(gam_car, col = "red", lwd = 3)
