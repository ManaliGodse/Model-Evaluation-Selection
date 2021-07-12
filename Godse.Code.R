

#installing libraries
install.packages('caret')
library(caret)
install.packages('Metrics')
library(Metrics)

#reading the data
train <- load("Data.train.RData")
test <- load("Data.test.RData")

#splitting the training data
set.seed(123)
T <- createDataPartition(Data.train$Y, times = 1, p=0.80, list= FALSE)
train_split <- Data.train[T,]
test_split <- Data.train[-T,]


#Implementing model on split data
model1 <- glm(Y ~ poly(X1, 5) + poly(X2, 5) + poly(X3, 5) + poly(X4, 5) + poly(X5,5) + 
            poly(X6, 5) + poly(X7, 5) + poly(X8, 5) + poly(X9, 5) + 
            poly(X10, 5) + poly(X11, 5) + poly(X12, 5) + poly(X13, 5) + 
            poly(X14, 5) + poly(X15, 5) + poly(X16, 5) + poly(X17, 5) + 
            poly(X18, 5) + poly(X19, 5) + poly(X20, 5) + poly(X21, 5) + 
            poly(X22, 5) + poly(X23, 5) + poly(X24, 5) + poly(X25, 5) + 
            poly(X26, 5) + poly(X27, 5) + poly(X28, 5) + poly(X29, 5) + 
            poly(X30, 5) + poly(X31, 5) + poly(X32, 5) + poly(X33, 5) + 
            poly(X34, 5) + poly(X35, 5) + poly(X36, 5) + poly(X37, 5) + 
            poly(X38, 5) + poly(X39, 5) + poly(X40, 5) + poly(X41, 5) + 
            poly(X42, 5) + poly(X43, 5) + poly(X44, 5) + poly(X45, 5) + 
            poly(X46, 5) + poly(X47, 5) + poly(X48, 5) + poly(X49, 5) + 
            poly(X50, 5) + poly(X51, 5) + poly(X52, 5) + poly(X53, 5) + 
            poly(X54, 5) + poly(X55, 5) + poly(X56, 5) + poly(X57, 5) + 
            poly(X58, 5) + poly(X59, 5) + poly(X60, 5) + poly(X61, 5) + 
            poly(X62, 5) + poly(X63, 5) + poly(X64, 5) + poly(X65, 5) + 
            poly(X66, 5) + poly(X67, 5) + poly(X68, 5) + poly(X69, 5) + 
            poly(X70, 5) + poly(X71, 5) + poly(X72, 5) + poly(X73, 5) + 
            poly(X74, 5) + poly(X75, 5) + poly(X76, 5) + poly(X77, 5) + 
            poly(X78, 5) + poly(X79, 5) + poly(X80, 5) + poly(X81, 5) + 
            poly(X82, 5) + poly(X83, 5) + poly(X84, 5) + poly(X85, 5) + 
            poly(X86, 5) + poly(X87, 5) + poly(X88, 5) + poly(X89, 5) + 
            poly(X90, 5) + poly(X91, 5) + poly(X92, 5) + poly(X93, 5) + 
            poly(X94, 5) + poly(X95, 5) + poly(X96, 5) + poly(X97, 5) + 
            poly(X98, 5) + poly(X99, 5) + poly(X100, 5) + poly(X101, 5) + 
            poly(X102, 5) + poly(X103, 5) + poly(X104, 5) + poly(X105, 5) + 
            poly(X106, 5) + poly(X107, 5) + poly(X108, 5) + poly(X109, 5) + 
            poly(X110, 5) + poly(X111, 5) + poly(X112, 5) + poly(X113, 5) + 
            poly(X114, 5) + poly(X115, 5) + poly(X116, 5) + poly(X117, 5) + 
            poly(X118, 5) + poly(X119, 5) + poly(X120, 5) + poly(X121, 5) + 
            poly(X122, 5) + poly(X123, 5) + poly(X124, 5) + poly(X125, 5) + 
            poly(X126, 5) + poly(X127, 5) + poly(X128, 5) + poly(X129, 5) + 
            poly(X130, 5) + poly(X131, 5) + poly(X132, 5) + poly(X133, 5) + 
            poly(X134, 5) + poly(X135, 5) + poly(X136, 5) + poly(X137, 5) + 
            poly(X138, 5) + poly(X139, 5) + poly(X140, 5), data = train_split)
mse(test_split$Y, predict(model1, test_split))

#Performing Cross Validation

train.control <- trainControl(method = "cv", number = 10)
# Train the model
model <- train(Y ~ poly(X1, 5) + poly(X2, 5) + poly(X3, 5) + poly(X4, 5) + poly(X5,5) + 
                 poly(X6, 5) + poly(X7, 5) + poly(X8, 5) + poly(X9, 5) + 
                 poly(X10, 5) + poly(X11, 5) + poly(X12, 5) + poly(X13, 5) + 
                 poly(X14, 5) + poly(X15, 5) + poly(X16, 5) + poly(X17, 5) + 
                 poly(X18, 5) + poly(X19, 5) + poly(X20, 5) + poly(X21, 5) + 
                 poly(X22, 5) + poly(X23, 5) + poly(X24, 5) + poly(X25, 5) + 
                 poly(X26, 5) + poly(X27, 5) + poly(X28, 5) + poly(X29, 5) + 
                 poly(X30, 5) + poly(X31, 5) + poly(X32, 5) + poly(X33, 5) + 
                 poly(X34, 5) + poly(X35, 5) + poly(X36, 5) + poly(X37, 5) + 
                 poly(X38, 5) + poly(X39, 5) + poly(X40, 5) + poly(X41, 5) + 
                 poly(X42, 5) + poly(X43, 5) + poly(X44, 5) + poly(X45, 5) + 
                 poly(X46, 5) + poly(X47, 5) + poly(X48, 5) + poly(X49, 5) + 
                 poly(X50, 5) + poly(X51, 5) + poly(X52, 5) + poly(X53, 5) + 
                 poly(X54, 5) + poly(X55, 5) + poly(X56, 5) + poly(X57, 5) + 
                 poly(X58, 5) + poly(X59, 5) + poly(X60, 5) + poly(X61, 5) + 
                 poly(X62, 5) + poly(X63, 5) + poly(X64, 5) + poly(X65, 5) + 
                 poly(X66, 5) + poly(X67, 5) + poly(X68, 5) + poly(X69, 5) + 
                 poly(X70, 5) + poly(X71, 5) + poly(X72, 5) + poly(X73, 5) + 
                 poly(X74, 5) + poly(X75, 5) + poly(X76, 5) + poly(X77, 5) + 
                 poly(X78, 5) + poly(X79, 5) + poly(X80, 5) + poly(X81, 5) + 
                 poly(X82, 5) + poly(X83, 5) + poly(X84, 5) + poly(X85, 5) + 
                 poly(X86, 5) + poly(X87, 5) + poly(X88, 5) + poly(X89, 5) + 
                 poly(X90, 5) + poly(X91, 5) + poly(X92, 5) + poly(X93, 5) + 
                 poly(X94, 5) + poly(X95, 5) + poly(X96, 5) + poly(X97, 5) + 
                 poly(X98, 5) + poly(X99, 5) + poly(X100, 5) + poly(X101, 5) + 
                 poly(X102, 5) + poly(X103, 5) + poly(X104, 5) + poly(X105, 5) + 
                 poly(X106, 5) + poly(X107, 5) + poly(X108, 5) + poly(X109, 5) + 
                 poly(X110, 5) + poly(X111, 5) + poly(X112, 5) + poly(X113, 5) + 
                 poly(X114, 5) + poly(X115, 5) + poly(X116, 5) + poly(X117, 5) + 
                 poly(X118, 5) + poly(X119, 5) + poly(X120, 5) + poly(X121, 5) + 
                 poly(X122, 5) + poly(X123, 5) + poly(X124, 5) + poly(X125, 5) + 
                 poly(X126, 5) + poly(X127, 5) + poly(X128, 5) + poly(X129, 5) + 
                 poly(X130, 5) + poly(X131, 5) + poly(X132, 5) + poly(X133, 5) + 
                 poly(X134, 5) + poly(X135, 5) + poly(X136, 5) + poly(X137, 5) + 
                 poly(X138, 5) + poly(X139, 5) + poly(X140, 5), data = train_split, method = "glm",
               trControl = train.control)

# Summarize the results
print(model)
model$resample
mse(test_split$Y, predict(model, test_split))
mean(model$resample$RMSE)


lastNameAndMSE = list(c('Godse',7.305))
preds <- predict(model, Data.test)
predsDf <- as.data.frame(preds)
results <- rbind(lastNameAndMSE, predsDf)
#head(results)
write.table( results, "Godse.Predictions.csv", sep=",", row.names = FALSE, col.names=FALSE)

