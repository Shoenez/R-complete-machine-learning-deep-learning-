library(MASS)
install.packages("pastecs")
pastecs::stat.desc(Cars93)

quantile(Cars93$Price, 0.95)
