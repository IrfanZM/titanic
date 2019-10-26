################################################################################
# File looking at the initial data provided
# Cleaning and feature engineering to maximise
# utility before running regressions
################################################################################

library(tidyverse)
library(MASS)

train <- read.csv("./data/train.csv")
test <- read.csv("./data/test.csv")

attach(train)

###########################################################
# Let's try a simple logistic regression to start off with
# with minimal transformation of variables
###########################################################

str(train)
# 891 observations, 12 variables

# Pclass
train %>% count(Pclass)
# all variables filled in, no imputation needed, but is filled in
# as integer - need to translate into factor
train$Pclass <- as.factor(train$Pclass)

# Leave name for now

# Sex
train %>% count(Sex)
# all looks sensible

# Age
age_dist <- train %>% 
  count(Age)
# 177 entries are NA, let's investigate these further
age_na_entries <- train %>%
  filter(is.na(Age))
# use a linear model to try and predict age from other characteristics
lm.age.fit <- lm(Age~Parch+SibSp+Fare,
                 data=train,
                 subset=!is.na(Age))
train <- train %>%
  mutate(age_imputed = round(predict(lm.age.fit,train),0)) %>%
  mutate(age_imputed = ifelse(age_imputed<=0,1,age_imputed)) %>%
  mutate(age_comb = ifelse(is.na(Age),age_imputed,Age))


# SibSp - number of siblings or spouses aboard the ship
train %>% count(SibSp)

# ParCh - number of parents or children aboard the ship
# can we assume that there are no grandparents? probably not 
# a safe assumption...
train %>% count(Parch)

# I think we're ready to run our linear model
# run on training set first
glm.survive <- glm(Survived~Pclass+Sex+age_comb+SibSp,
                         data=train,
                         family="binomial")
glm.survive.probs <- predict(glm.survive, type="response")
glm.survive.preds <- round(glm.survive.probs,0)
table(glm.survive.preds,train$Survived)
mean(glm.survive.preds==train$Survived)
# 79% success prediction rate
# let's compare to just women predictor
table(train$Sex,train$Survived)
mean(ifelse(train$Sex=="female",1,0)==train$Survived)
# 78% lol. I think I need different models for both males and females...

##############################################
# Run on test 
##############################################
test$Pclass <- as.factor(test$Pclass)

test <- test %>%
  mutate(age_imputed = round(predict(lm.age.fit,test),0)) %>%
  mutate(age_imputed = ifelse(age_imputed<=0,1,age_imputed)) %>%
  mutate(age_comb = ifelse(is.na(Age),age_imputed,Age))

test.probs <- predict(glm.survive,
                      test,
                      type = "response")
test.preds <- round(test.probs,0)
output <- data.frame(test$PassengerId, test.preds)
names(output) <- c("PassengerId","Survived")
write.csv(output,"output_log1.csv", row.names = FALSE)
