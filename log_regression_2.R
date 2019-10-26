################################################################################
# File looking at the initial data provided
# Cleaning and feature engineering to maximise
# utility before running regressions
#
# And then multiple methods are used to try and obtain best outcome
################################################################################

#install.packages("tidyverse")

library(MASS)
library(dplyr)
library(purrr)
library(stringr)
library(mice)
library(class)
library(caret)

train <- read.csv("./data/train.csv") %>% as_tibble()


attach(train)

###########################################################
# Let's try a simple logistic regression to start off with
# more in-depth transformation of variables
###########################################################

str(train)
# 891 observations, 12 variables

# Idenftify missing variables
md.pattern(train)


# Pclass
train %>% count(Pclass)
# all variables filled in, no imputation needed, but is filled in
# as integer - need to translate into factor
train$Pclass <- as.factor(train$Pclass)

# Obtain title. If title occurence <8, than assign as 'rare'
# Regex to get title
train$title <- str_extract(Name,"(?<=,\\s)[:alpha:]+")
# Attach title onto original dataset
title_occurence <- count(train,title) %>% 
  arrange(desc(n))
train <- left_join(train,title_occurence,by="title") %>%
  mutate(title_agg = ifelse(n<7,"rare",title))
# visualise effect of title on survivability
ggplot(train, aes(title_agg,as.factor(Survived))) + geom_count()

# Sex
train %>% count(Sex)
# all looks sensible

# Age
# Let's look at distribution against survivability
ggplot(train,aes(as.factor(Survived),Age)) + geom_violin()
# Looks like there's a slight advantage to being a child, 
# otherwise no real difference
age_dist <- train %>% 
  count(Age)
# 177 entries are NA
# visualise missing data with mice
md.pattern(train)
# Age is the only missing value (except cabin, but not too useful)
# Use multiple imputation to get age, first getting dependent columns
train.age <- dplyr::select(train,Pclass,Sex,SibSp,Parch,Age)
age.imputed <- mice(train.age,m=5,maxit=50,method='pmm',seed=500)
# take first imputed data set
train$age_imputed <- complete(age.imputed)$Age

# Add marker for if there is a cabin
train$has_cabin <- ifelse(Cabin=="",0,1)

# try splitting into age brackets
# -- Turns out this reduces predictive accuracy on test model!
# -- could be due to variance?
age_bracket_function <- function(age) {
  if (is.na(age)==TRUE) {
    bracket = "unknown"
  }
  else if (age<=10) {
    bracket = "under 10"
  }
  else if (age>10 & age <=18) {
    bracket = "11-18"
  }
  else if (age>18 & age <=25) {
    bracket = "19-25"
  }
  else if (age>25 & age <=40) {
    bracket = "25-40"
  }
  else if (age>40 & age <=60) {
    bracket = "40-60"
  }
  else {bracket = "over 60"}
  return(bracket)
}
#train$age_bracket <- sapply(train$age_imputed,age_bracket_function)

##############################################
# Clean test data
##############################################
test <- read.csv("./data/test.csv") %>% as_tibble()
attach(test)
test$Pclass <- as.factor(test$Pclass)

test$title <- str_extract(Name,"(?<=,\\s)[:alpha:]+")
title_occurence <- count(test,title) %>% 
  arrange(desc(n))
test <- left_join(test,title_occurence,by="title") %>%
  mutate(title_agg = ifelse(n<8,"rare",title))
md.pattern(test)

test.age <- dplyr::select(test,Pclass,Sex,SibSp,Parch,Age,Fare,Embarked)
age.imputed <- mice(test.age,m=5,maxit=50,method='pmm',seed=500)
test$age_imputed <- complete(age.imputed)$Age
test$Fare <- complete(age.imputed)$Fare
test$has_cabin <- ifelse(Cabin=="",0,1)


##############################################
# Train model
##############################################

# General logistic regression
glm.survive <- glm(Survived~Pclass+age_imputed+Sex+SibSp+Parch+title_agg+has_cabin,
                   data=train,
                   family="binomial")
glm.survive.probs <- predict(glm.survive, type="response")
glm.survive.preds <- round(glm.survive.probs,0)
table(glm.survive.preds,train$Survived)
mean(glm.survive.preds==train$Survived)

# For fun, let's see how well a female=survive model does
table(train$Sex,train$Survived)
mean(ifelse(train$Sex=="female",1,0)==train$Survived)
# 78% lol. I think I need different models for both males and females...


##############################################
# Run on test data
##############################################

# General logistic regression
test.probs <- predict(glm.survive,
                      test,
                      type = "response")
test.preds <- round(test.probs,0)
output <- data.frame(test$PassengerId, test.preds)
names(output) <- c("PassengerId","Survived")
write.csv(output,"output_log4.csv", row.names = FALSE)
