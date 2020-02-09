#
#Feature Selection with the Caret R Package
##https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

#load tidyverse to use read_csv
library(tidyverse)
library(caret)
library(mlbench)

training<-read_csv("C:/Users/jccos/Documents/pml-training.csv")


#load only numerical features relevant to quality measurements

train1<-select(training,
              max_roll_belt:max_picth_belt,
              total_accel_belt,
              amplitude_roll_belt:amplitude_pitch_belt,
              var_total_accel_belt:magnet_arm_z,
              min_roll_belt:min_pitch_belt,
              max_roll_arm:yaw_dumbbell,
              max_roll_dumbbell:amplitude_pitch_dumbbell,
              total_accel_dumbbell:yaw_forearm,
              max_roll_forearm:max_picth_forearm,
              min_roll_forearm:min_pitch_forearm,
              amplitude_roll_forearm:amplitude_pitch_forearm,
              total_accel_forearm:magnet_forearm_z,
              classe)

train2<-select(train1,total_accel_belt,
               gyros_belt_x:total_accel_arm,
               gyros_arm_x:magnet_arm_z,
               roll_dumbbell:yaw_dumbbell,
               total_accel_dumbbell,
               gyros_dumbbell_x:yaw_forearm,
               total_accel_forearm,
               gyros_forearm_x:magnet_forearm_z,classe)
               

colSums(is.na(train2))

#preprocessing covariates; loooking only on numerical predictable variables.

M<-abs(cor(train1[,-120],use = "pairwise.complete.obs"))
diag(M)<-0
which(M>0.8,arr.in=T)

#Looking at the data we see several variable that are highly correlated
#with each other, so probably a good idea to perform Principal Component
#Analyis to reduce number of predictors and reduce some fo the noise


preProc2<-preProcess(train2[,-50],method="pca",pcaComp = 10)

trainPC2<-predict(preProc2,train2[,-50])

plot(trainPC2$x[,1],trainPC2$x[,2])

modelFit<-train(train2$classe ~.,method="rf",data=trainPC2)


M<-abs(cor(train2[,-50],use = "pairwise.complete.obs"))
diag(M)<-0
temp<-which(M>0.8,arr.in=T)


prComp2<-prcomp(train2[,-50],scale=TRUE)
prComp2_sd<-prComp2$sdev
prComp2_var<-prComp2_sd^2
prCom2_var_ex<-prComp2_var/sum(prComp2_var)
sum(prCom2_var_ex[1:15])

plot(cumsum(prCom2_var_ex),xlab="Principal Component",
     ylab="Cumulative Proportion of Variance Explained")

### BUILDING THE MODEL
train2.pr<-data.frame(classe=train2$classe,prComp2$x)
train2.pr.new<-train2.pr[,1:15]
modelfit<-train(classe~.,data=train2.pr.new,method="rf")

modelpred<-predict(modelfit,newdata=train2.pr.new)
confusionMatrix(modelpred,train2.pr.new$classe)


##Test Data

testing<-read_csv("C:/Users/jccos/Documents/pml-testing.csv")
test1<-select(testing,
               max_roll_belt:max_picth_belt,
               total_accel_belt,
               amplitude_roll_belt:amplitude_pitch_belt,
               var_total_accel_belt:magnet_arm_z,
               min_roll_belt:min_pitch_belt,
               max_roll_arm:yaw_dumbbell,
               max_roll_dumbbell:amplitude_pitch_dumbbell,
               total_accel_dumbbell:yaw_forearm,
               max_roll_forearm:max_picth_forearm,
               min_roll_forearm:min_pitch_forearm,
               amplitude_roll_forearm:amplitude_pitch_forearm,
               total_accel_forearm:magnet_forearm_z
               )

test2<-select(test1,total_accel_belt,
               gyros_belt_x:total_accel_arm,
               gyros_arm_x:magnet_arm_z,
               roll_dumbbell:yaw_dumbbell,
               total_accel_dumbbell,
               gyros_dumbbell_x:yaw_forearm,
               total_accel_forearm,
               gyros_forearm_x:magnet_forearm_z)

test.pr<-predict(prComp2, newdata = test2)
test.pr <- as.data.frame(test.pr)
test.pr.2 <- test.pr[,1:15]
pred.test2 <- predict(modelfit, test.pr.2)
pred.test2
