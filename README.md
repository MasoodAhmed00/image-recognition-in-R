# image-recognition-in-R
image recognition of cat and dog in R programming using keras and tensorflow
# Install packages

#install.packages("keras")
#install.packages("BiocManager") 
#BiocManager::install("EBImage")


#Load Packages
library(tidyverse)
library(EBImage)
require(caret)
library(keras)


#Read images
setwd('C:\\Users\\Star\\Downloads\\cat and dog')
pics <- c('c1.jpeg','c2.jpeg','c3.jpeg','c4.jpeg','c5.jpeg','c6.jpeg',
          'd1.jpeg','d2.jpeg','d3.jpeg','d4.jpeg','d5.jpeg','d6.jpeg')

my_pics <- list()
for(i in 1:12){my_pics[[i]] <- readImage(pics[i])}

#Explore
print(my_pics[[1]])
display(my_pics[[1]])
summary(my_pics[[1]])
hist(my_pics[[1]])
str(my_pics)

#resize
for(i in 1:12){my_pics[[i]] <- resize(my_pics[[i]],28,28)}

#reshape
for(i in 1:12){my_pics[[i]] <- array_reshape(my_pics[[i]],c(28,28,3))}

#Row bind
train_x <- NULL
for(i in 1:5){train_x <- rbind(train_x,my_pics[[i]])}
for(i in 7:11){train_x <- rbind(train_x,my_pics[[i]])}
str(train_x)

test_x <- rbind(my_pics[[6]],my_pics[[12]])
train_y <- c(0,0,0,0,0,1,1,1,1,1)
test_y <- c(0,1)




library(reticulate)
library(tensorflow)

reticulate::virtualenv_create(install_tensorflow())
reticulate::virtualenv_remove()


# One Hot Encoding
train_labels <- to_categorical(train_y)
test_labels <- to_categorical(test_y)

#Model
model <- keras_model_sequential()
model %>% 
  layer_dense(units=256,activation = 'relu',input_shape = c(2352)) %>% 
  layer_dense(units=128,activation = 'relu') %>% 
  layer_dense(units=2,activation = 'softmax')
summary(model)


#Compile
model %>% 
  compile(loss = 'binary_crossentropy',
          optimizer = optimizer_rmsprop(),
          metrics = c('accuracy'))



#Fit model
history <- model %>% 
  fit(train_x,
      train_labels,
      epochs = 30,
      batch_size = 32,
      validation_split = 0.2)


#Modeel Evaluation and predictions of train data
model %>%  evaluate(train_x,train_labels)
#pred <- model %>% predict_classes(train_x)
pred <- model %>% predict(train_x) %>% k_argmax()


#Modeel Evaluation and predictions of train data
model %>% evaluate(test_x,test_labels)



















