# kaggle machine learning class

library(tidyverse)
library(reshape2)

setwd("~/R/machinelearning_housing/handson-ml-master/datasets/housing")
housing <- read.csv("housing.csv")

View(housing)
summary(housing)

par(mfrow=c(2,5))
colnames(housing)

ggplot(data = melt(housing), mapping = aes(
  x = value)) + 
  geom_histogram(bins = 30) + 
  facet_wrap(~variable, scales = 'free_x')

# to move forward, we need to impute the missing values 
# in 'total_bedrooms'

housing$total_bedrooms[is.na(housing$total_bedrooms)] = 
  median(housing$total_bedrooms, na.rm = TRUE)

# now, we should make the 'total_bedrooms' and 
# 'total_rooms' in each community and make them into
# means using the number of households column value

housing$mean_bedrooms = housing$total_bedrooms/
  housing$households
housing$mean_rooms = housing$total_rooms/
  housing$households

drops = c('total_bedrooms', 'total_rooms')
housing = housing[, !(names(housing) %in% drops)]

# turning categorical variables into boolean values

cat_housing <- housing %>% 
  # select column 'ocean proximity' for cat_housing
  select(ocean_proximity) %>% 
  # create columns 'val' and 'ID' 
  # (R needs a unique identifier for an anchor when pivoting)
  mutate(val = 1, ID = rownames(.)) %>% 
  # split variables into columns and replace NA with 0
  pivot_wider(names_from = ocean_proximity, values_from = val,
              values_fill = list(val = 0)) %>%  
  select(-ID)

View(cat_housing)

#since some numerical values have drastically different scales,
#these steps will help make them more relative

drops = c('ocean_proximity', 'median_house_value', 'ID')
housing_num = housing[ , !(names(housing) %in% drops)]
scaled_housing_num = scale(housing_num)

# now to merge the altered numerical and categorical dataframes

cleaned_housing <- cbind(cat_housing, scaled_housing_num, 
                         median_house_value = housing$median_house_value)

# now for the nitty gritty - creating training and testing datasets

set.seed(1738)
sample <- sample.int(n = nrow(cleaned_housing),
                     size = floor(.8*nrow(cleaned_housing)),
                     replace = F)
train = cleaned_housing[sample, ] #just sample data
test = cleaned_housing[-sample, ] #all data except sample

#check to make sure that the training data is random (the index should
# be jumbled), and that the number of observations between the two sets
# adds up to the original dataset

head(train)
nrow(train) + nrow(test) == nrow(cleaned_housing)

#now to test some predictive models!
#we're using a generalized linear model with 5 folds first

library('boot')
glm_house <- glm(median_house_value ~ median_income + 
                   mean_rooms + population, data = 
                   cleaned_housing)
k_fold_cv_error <- cv.glm(cleaned_housing, glm_house, K=5)
k_fold_cv_error$delta #The first component is the raw cross-validation estimate of 
                      #prediction error. The second component is the adjusted 
                      #cross-validation estimate.

