###################################################
################ Exercise 1 ######################
##################################################

#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
library(party)
library(rpart.plot)

#Load data
T=read.csv('./Data/Actors.csv')
#Fit Tree
fitTree=rpart(as.factor(IMDb)~Actor+Budget, data=T, method="anova",control =rpart.control(minsplit =1,minbucket=1, cp=0))

#Plot the tree
prp(fitTree)					
prp(fitTree,varlen=3)


printcp(fitTree) # display the results 
plotcp(fitTree) # visualize cross-validation results 
summary(fitTree) # detailed summary of splits
