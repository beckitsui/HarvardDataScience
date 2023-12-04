# # # MovieLense Machine Learning Project - SVD 

#load librarys
library(tibble)
library(readr)
library(dplyr)
library(tidyr)
library(recommenderlab)

# create a RMSE function to avoid errors caused by NA
# also slightly less computationally intensive...
RMSE <- function(true, predict) {
  differences <- true - predict
  differences <- differences[!is.na(differences)]
  sqrt(mean(differences^2))
}

set.seed(2023)
# too bad I don't have more powerful RAMs, can edx sponsor me some Nvidia RAM 
#next time it asks me to do ml project with 8million rows of data?
# this is about the max. number of movielens data my computer has memory to 
# process for pivot table and matrix, after many hours of testing + sweat & tears 
maxRowNum = 170000 
# create a subset from edx df to perform testing and training
edx_small <- sample_n(edx, size = maxRowNum/0.8)

test_ind <- createDataPartition(edx_small$userId, times = 1, p=0.2, list = FALSE)
test_subset <- edx_small[test_ind,]
train_subset <- edx_small[-test_ind,]

#clean test and train set
test_subset <- test_subset |>
  semi_join(train_subset, by = "movieId") |>
  semi_join(train_subset, by = "userId") 
train_subset <- train_subset |>
  semi_join(test_subset, by = "movieId") |>
  semi_join(test_subset, by = "userId") 

#save and load test and train set to save my computer from running out of memory......
#write.csv(train_subset, "train_subset.csv", row.names = FALSE)
#write.csv(test_subset, "test_subset.csv", row.names = FALSE)
#train_subset <- read.csv("train_subset.csv")
#test_subset <- read.csv("test_subset.csv")

rm(edx, edx_small)


# Transform the train_subset into matrix
# pivot the df to create matrix: user a row, movie as column
rating_matrix <- train_subset |>
  select(movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) |>
  column_to_rownames(var = "userId")

# convert to matrix
rating_matrix <- as.matrix(rating_matrix)

# calculate all movie avg
mu <- mean(rating_matrix, na.rm = TRUE)
n <- colSums(!is.na(rating_matrix))
sums <- colSums(rating_matrix - mu, na.rm = TRUE)

#create a fit movie data frame for lambda testing 
fit_movies <- data.frame(movieId = as.integer(colnames(rating_matrix)),
                         mu = mu,
                         b_i_reg = 0 )


## Regularization
# calculate the movie effect b_i_reg & find the estimated user effect b_u 

# for movie effect, finding lambda to penalize movie with few rating 
lambdas = seq(0,10,.1)
lambdas_rmse <- sapply(lambdas, function(lambda){
  b_i <- sums/(n + lambda)
  fit_movies$b_i <-  b_i
  #left join to the test set
  left_join(test_subset, fit_movies, by='movieId') |>
    #create predicted value by adding mu + b_i
    mutate(pred = mu + b_i) |>
    #calculate RMSE 
    summarize(rmse = RMSE(rating, pred)) |>
    pull(rmse)
})
#the lambda value that minimize RMSE is 2.3
lambda <- lambdas[which.min(lambdas_rmse)]


#remove the user effects and movie effects from the matrix 

b_i_reg <- colSums(rating_matrix - mu, na.rm=TRUE)/(n + lambda)
fit_movies$b_i_reg <- b_i_reg
fit_users <- data.frame(userId = as.integer(rownames(rating_matrix)), 
                        b_u = rowMeans(sweep(rating_matrix-mu,2,b_i_reg), na.rm=TRUE))

rating_matrix <- sweep(rating_matrix-mu, 2, fit_movies$b_i_reg) - fit_users$b_u
#rating_matrix[1:10,1:10]

# replace NA with 0 or -1 for the svd model,
# -1 assume user dislikes the movie or movie is unpopular
rating_matrix[is.na(rating_matrix)] <- 0
#rating_matrix[1:10, 1:10] 


## SVD Model Training 

svd_model <- svd(rating_matrix)
#save(svd_model, file = "svd_model.RData")
#load("svd_model.RData")

#After running the svd model, I am constructing an approximated matrix
#I only select 10 features to avoid overfitting 

#user
U <- svd_model$u
#movie
V <- svd_model$v
Sigma <- diag(svd_model$d)

#assign row names and column names for the user and movie matrix U and V
rownames(U) <- rownames(rating_matrix)  # User IDs as row names
colnames(V) <- colnames(rating_matrix)  # Movie IDs as column names

# select number of features to use, avoid overfitting
num_features <- 10

# reconstruct the rating matrix using a subset of features
# U[, 1:num_features] %*% 
# Sigma[1:num_features, 1:num_features] %*% 
# t(V[, 1:num_features])
approx_rating_matrix <- 
  U[, 1:num_features] %*% Sigma[1:num_features, 1:num_features] %*% t(V[, 1:num_features])

# set the proper row and column names
rownames(approx_rating_matrix) <- rownames(rating_matrix)
colnames(approx_rating_matrix) <- colnames(rating_matrix)

## Testing on test set
#Need to transform the test set so that the test rating matrix is in the same structure as the approx_rating_matrix
#Also transform the approx_rating_matrix to contain only user and movie in test set

# test set pivot table transformation and convert to matrix 
test_rating_matrix <- test_subset  |>
  select(movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) |>
  column_to_rownames(var = "userId")
test_rating_matrix <- as.matrix(test_rating_matrix)

# replace the NA with 0 in the matrix
test_rating_matrix[is.na(test_rating_matrix)] <- 0

# find the common colnames and rownames between approx. matrix and test matrix
common_users <- intersect(row.names(test_rating_matrix), row.names(approx_rating_matrix))
common_movies <- intersect(colnames(test_rating_matrix), colnames(approx_rating_matrix))

# transform the approx_rating_matrix to contain only user and movie in test set
aligned_approx_ratings <- approx_rating_matrix[common_users, common_movies]
aligned_test_ratings <- test_rating_matrix[common_users, common_movies]

#remove unnecessary large matrix to save some memory space...
rm(test_rating_matrix)


# Function to calculate RMSE in chunks
calculate_rmse_in_chunks <- function(actual, predicted, chunk_size) {
  n <- nrow(actual)
  rmse_values <- numeric(ceiling(n / chunk_size))
  
  for (i in seq(1, n, by = chunk_size)) {
    chunk_end <- min(i + chunk_size - 1, n)
    chunk_actual <- actual[i:chunk_end, , drop = FALSE]
    chunk_predicted <- predicted[i:chunk_end, , drop = FALSE]
    rmse_values[ceiling(i / chunk_size)] <- RMSE(chunk_actual, chunk_predicted)
  }
  
  mean(rmse_values, na.rm = TRUE)
}

# Calculate RMSE in chunks
chunk_size <- 10000 

#add mu back to the approx. rating 

#aligned_approx_ratings_withMU <- aligned_approx_ratings + mu
#aligned_test_ratings_withMU <- aligned_test_ratings - mu

rmse <- calculate_rmse_in_chunks(aligned_test_ratings, aligned_approx_ratings, chunk_size)

#aligned_test_ratings[1:10, 1:10] 
rmse

#The rmse for the test set before regularization is 0.07508595
#The rmse for the test set after regularization is 0.07195849


  ## Testing on final_hold_out set (last required step)
# Need to transform the final hold out set so that the test rating matrix is in the same structure as the approx_rating_matrix
# Also transform the approx_rating_matrix to contain only user and movie in final hold out set

# I am creating a small subset of final hold out since my model can take around 170,000 
# rows of movielens data at max.

final_holdout_test_small <- final_holdout_test |>
  semi_join(train_subset, by = "movieId") |>
  semi_join(train_subset, by = "userId") |>
  sample_n(size = maxRowNum)

head(final_holdout_test_small)

#remove unnecessary large matrix to save some memory space  
#otherwise, my code can not run...
rm(rating_matrix, svd_model, aligned_test_ratings)

# test set pivot table transformation and convert to matrix 
finaltest_rating_matrix <- final_holdout_test_small  |>
  select(movieId, userId, rating) |>
  pivot_wider(names_from = movieId, values_from = rating) |>
  column_to_rownames(var = "userId")
finaltest_rating_matrix <- as.matrix(finaltest_rating_matrix)

# replace the NA with 0 in the matrix
finaltest_rating_matrix[is.na(finaltest_rating_matrix)] <- 0

# find the common colnames and rownames between approx. matrix and test matrix
common_users <- 
  intersect(row.names(finaltest_rating_matrix), row.names(approx_rating_matrix))
common_movies <- 
  intersect(colnames(finaltest_rating_matrix), colnames(approx_rating_matrix))

# transform the approx_rating_matrix to contain only user and movie in test set
aligned_approx_ratings <- approx_rating_matrix[common_users, common_movies]
aligned_finaltest_ratings <- finaltest_rating_matrix[common_users, common_movies]

#### FINAL STEP: Calculating the rmse with svd model on final hold out set
rmse_final <- calculate_rmse_in_chunks(aligned_finaltest_ratings, 
                                       aligned_approx_ratings, 
                                       chunk_size)

#final RMSE 
print(rmse_final)
