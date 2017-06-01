# This is an example of IV estimation of The Effect of Institutions on Economic Growth using Double Machine Learning Methods 
# References: "Double/Debiased Machine Learning of Treatment and Causal Parameters",  AER P&P 2017     
#             "Double Machine Learning for Treatment and Causal Parameters",  Arxiv 2016 

# This empirical example uses the data from Acemoglu, D., Johnson, S., Robinson, J.A., 2001. The Colonial Origins of Comparative Development: An 
# Empirical Investigation. American Economic Review 91 (5), 1369–1401
#######################################################################################################################################################

###################### Loading packages ###########################

library(foreign);
library(quantreg);
library(mnormt);
library(gbm);
library(glmnet);
library(MASS);
library(rpart);
library(doParallel)
library(sandwich);
library(hdm);
library(randomForest);
library(nnet)
library(matrixStats)
library(quadprog)
library(xtable)
library(ivmodel)

###################### Loading functions and Data ##############################

rm(list = ls())  # Clear everything out so we're starting clean
source("ML_Functions.R")  
source("Moment_Functions.R")  
options(warn=-1)
set.seed(1211);
cl   <- makeCluster(12, outfile="")

data(AJR); 
data <- AJR

################################ Inputs ########################################

# Outcome Variable
y  <- "GDP"

# Treatment Variable
d  <- "Exprop"

# Instrument
z  <- "logMort"	

# Controls
x  <- "Latitude + Latitude2 + Africa +  Asia + Namer + Samer"       # use this for tree-based methods like forests and boosted trees
xl <- "(Latitude + Latitude2 + Africa +  Asia + Namer + Samer)^2"   # use this for rlasso etc.


# Method names: Boosting, Nnet, RLasso, PostRLasso, Forest, Trees, Ridge, Lasso, Elnet, Ensemble

Boosting     <- list(n.minobsinnode = 1, bag.fraction = .5, train.fraction = 1.0, interaction.depth=2, n.trees=1000, shrinkage=.01, n.cores=1, cv.folds=2, verbose = FALSE, clas_dist= 'adaboost', reg_dist='gaussian')
Forest       <- list(clas_nodesize=3, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
RLasso       <- list(intercept = TRUE)
Nnet         <- list(size=2,  maxit=1000, decay=0.01, MaxNWts=10000,  trace=FALSE)
Trees        <- list(reg_method="anova", clas_method="class")

arguments    <- list(Boosting=Boosting, Forest=Forest, RLasso=RLasso, Nnet=Nnet, Trees=Trees)

ensemble     <- list(methods=c("RLasso", "Boosting", "Forest"))                       # methods for the ensemble estimation
methods      <- c("RLasso","Trees", "Boosting", "Forest", "Nnet","Ensemble")          # ML methods that are used in estimation
split        <- 100                                                                   # number of splits

################################ Estimation ##################################################

############## Arguments for DoubleML function:

# data:     : data matrix
# y         : outcome variable
# d         : treatment variable
# z         : instrument
# xx        : controls for tree-based methods
# xL        : controls for penalized linear methods
# methods   : machine learning methods
# DML       : DML1 or DML2 estimation (DML1, DML2)
# nfold     : number of folds in cross fitting
# est       : estimation methods (IV, LATE, plinear, interactive)
# arguments : argument list for machine learning methods
# ensemble  : ML methods used ine ensemble method
# silent    : whether to print messages
# trim      : bounds for propensity score trimming


r <- foreach(k = 1:split, .combine='rbind', .inorder=FALSE, .packages=c('MASS','randomForest','neuralnet','gbm', 'sandwich', 'hdm', 'nnet', 'rpart','glmnet')) %dopar% { 
  
  dml <- DoubleML(data=data, y=y, d=d, z=z, xx=x, xL=xl, methods=methods, DML="DML2", nfold=2, est="IV", arguments=arguments, ensemble=ensemble, silent=FALSE, trim=NULL) 
  
  data.frame(t(dml[1,]), t(dml[2,]))
  
}

################################ Compute Output Table ########################################

result           <- matrix(0,3, length(methods)+1)
colnames(result) <- cbind(t(methods), "best")
rownames(result) <- cbind("Median ATE", "se(median)",  "se")

result[1,]        <- colQuantiles(r[,1:(length(methods)+1)], probs=0.5)
result[2,]        <- colQuantiles(sqrt(r[,(length(methods)+2):ncol(r)]^2+(r[,1:(length(methods)+1)] - colQuantiles(r[,1:(length(methods)+1)], probs=0.5))^2), probs=0.5)
result[3,]        <- colQuantiles(r[,(length(methods)+2):ncol(r)], probs=0.5)

result_table <- round(result, digits = 2)

for(i in 1:ncol(result_table)){
  for(j in seq(2,nrow(result_table),3)){
    
    result_table[j,i] <- paste("(", result_table[j,i], ")", sep="")
    
  }
  for(j in seq(3,nrow(result_table),3)){
    
    result_table[j,i] <- paste("(", result_table[j,i], ")", sep="")
    
  }
}

print(xtable(result_table, digits=3))

