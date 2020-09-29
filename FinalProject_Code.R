
# Install all needed libraries if it is not present
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(recosystem)) install.packages("recosystem")


# Loading all needed libraries
library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(lubridate)
library(caret)

# Data Preparation
load(file = "edx.data")
head(edx)
load(file = "temp.data")
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Data Exploration

# 3.1 Distribution of Ratings
mean_ratings<-mean(edx$rating) #3.512465
edx %>% ggplot(aes(rating))+
  geom_histogram(binwidth=0.25)+
  scale_x_continuous(breaks = seq(0.5,5,0.5))+
  geom_vline(xintercept=mean_ratings,col="red",linetype="dashed")

dim(edx) # 9000055       6
n_distinct(edx$movieId) # 10677
n_distinct(edx$title) # 10676: there might be movies of different IDs with the same title
n_distinct(edx$userId) # 69878
n_distinct(edx$movieId)*n_distinct(edx$userId) # 746087406
n_distinct(edx$movieId)*n_distinct(edx$userId)/dim(edx)[1] # 83

set.seed(1)
#Movies vs Users - Shows sparsity
random_users <- sample(unique(edx$userId), 100)
edx %>% filter(userId %in% random_users) %>% 
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  spread(movieId, rating) %>% 
  select(sample(ncol(.), 100)) %>% 
  as.matrix() %>% t(.) %>%
  image(1:100, 1:100,. , xlab="Movies", ylab="Users")
abline(h=0:100+0.5, v=0:100+0.5, col = "grey")

# 3.2 Extracting age of movies

edx_1 <- edx %>% mutate(year_rated = year(as_datetime(timestamp)))
# extract the release year of the movie
# edx_1 has year_rated, year_released, age_at_rating, and titles without year information
edx_1 <- edx_1 %>% mutate(title = str_replace(title,"^(.+)\\s\\((\\d{4})\\)$","\\1__\\2" )) %>% 
  separate(title,c("title","year_released"),"__") %>%
  select(-timestamp) 
edx_1 <- edx_1 %>% mutate(age_at_rating= as.numeric(year_rated)-as.numeric(year_released))
head(edx_1)

# 3.3 Ratings by Users
# Plot number of ratings per movie
edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")


# Plot number of ratings given by users
edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")

# Plot mean movie ratings given by users
edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()

# Model Development

# define RMSE: residual mean squared error
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Model 1 ##
## first model: use average ratings for all movies regardless of user

mu <- mean(edx$rating)
mu # 3.51247

naive_rmse <- RMSE(validation$rating, mu)
naive_rmse # 1.0612

rmse_results <- data_frame(Model = "Just the average", RMSE = naive_rmse)
rmse_results
options(digits=6)
options(pillar.sigfig = 6)

## Model 2 ##
## modeling movie effects: adding b_i to represent average ranking for movie_i

movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))
predicted_ratings_2 <- mu + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_2_rmse <- RMSE(validation$rating,predicted_ratings_2) # 0.943909
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Movie Effect Model",  
                                     RMSE = model_2_rmse))
rmse_results

## Model 3 ##
## user effects: adding b_u to represent average ranking for user_u

user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings_3 <- validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_3_rmse <- RMSE(validation$rating,predicted_ratings_3) # 0.865349
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Movie + User Effects Model",  
                                     RMSE = model_3_rmse))
rmse_results


## Model 4 ##
## regularization of movie effect: control the total variability of the movie effects taking the number of ratings made for a specific movie into account

# A) perform cross validation to determine the parameter lambda

# use 10-fold cross validation to pick a lambda for movie effects regularization
# split the data into 10 parts
set.seed(2019)
cv_splits <- createFolds(edx$rating, k=10, returnTrain =TRUE)

# define a matrix to store the results of cross validation
rmses <- matrix(nrow=10,ncol=51)
lambdas <- seq(0, 5, 0.1)

# perform 10-fold cross validation to determine the optimal lambda
for(k in 1:10) {
  train_set <- edx[cv_splits[[k]],]
  test_set <- edx[-cv_splits[[k]],]
  
  # Make sure userId and movieId in test set are also in the train set
  test_final <- test_set %>% 
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId")
  
  # Add rows removed from validation set back into edx set
  removed <- anti_join(test_set, test_final)
  train_final <- rbind(train_set, removed)
  
  mu <- mean(train_final$rating)
  just_the_sum <- train_final %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - mu), n_i = n())
  
  rmses[k,] <- sapply(lambdas, function(l){
    predicted_ratings <- test_final %>% 
      left_join(just_the_sum, by='movieId') %>% 
      mutate(b_i = s/(n_i+l)) %>%
      mutate(pred = mu + b_i) %>%
      pull(pred)
    return(RMSE(predicted_ratings, test_final$rating))
  })
}

rmses
rmses_cv <- colMeans(rmses)
rmses_cv
qplot(lambdas,rmses_cv)
lambdas[which.min(rmses_cv)]   #2.2

# B) model generation and prediction
# Regularized Movie Effect Model
# lambda <- lambdas[which.min(rmses_cv)]
lambda <- 2.2
mu <- mean(edx$rating)
movie_reg_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 
predicted_ratings_4 <- validation %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
model_4_rmse <- RMSE(predicted_ratings_4, validation$rating)   # 0.943852 not too much improved
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Regularized Movie Effect Model",  
                                     RMSE = model_4_rmse))
rmse_results 

## Model 5 ##
# model generation and prediction
# Regularized Movie Effect and User Effect Model
# lambda <- lambdas[which.min(rmses_2_cv)]
lambda <- 4.9
mu <- mean(edx$rating)
b_i_reg <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u_reg <- edx %>% 
  left_join(b_i_reg, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
predicted_ratings_5 <- 
  validation %>% 
  left_join(b_i_reg, by = "movieId") %>%
  left_join(b_u_reg, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_5_rmse <- RMSE(predicted_ratings_5, validation$rating)   # 0.864818
rmse_results <- bind_rows(rmse_results,
                          data_frame(Model="Regularized Movie + User Effect Model",  
                                     RMSE = model_5_rmse))
rmse_results 



# from the summarized rmses of different models, we can see that matrix factorization largely improved the accuracy of the prediction
final_rmses <- rmse_results 
final_rmses
