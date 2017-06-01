#######################################################################################################################################################
#  This is an example of ATE estimation of unemployment bonus on duration using Double Machine Learning Methods 
#  References: "Double/Debiased Machine Learning of Treatment and Causal Parameters",  AER P&P 2017     
#              "Double Machine Learning for Treatment and Causal Parameters",  Arxiv 2016               

# Data source: Yannis Bilias, "Sequential Testing of Duration Data: The Case of Pennsylvania 'Reemployment Bonus' Experiment", 
# Journal of Applied Econometrics, Vol. 15, No. 6, 2000, pp. 575-594

# Description of the data set taken from Bilias (2000):

# The 23 variables (columns) of the datafile utilized in the article may be described as follows:

# abdt:       chronological time of enrollment of each claimant in the Pennsylvania reemployment bonus experiment.
# tg:         indicates the treatment group (bonus amount - qualification period) of each claimant. 
# inuidur1:   a measure of length (in weeks) of the first spell ofunemployment
# inuidur2:   a second measure for the length (in weeks) of 
# female:     dummy variable; it indicates if the claimant's sex is female (=1) or male (=0).
# black:      dummy variable; it  indicates a person of black race (=1).
# hispanic:   dummy variable; it  indicates a person of hispanic race (=1).
# othrace:    dummy variable; it  indicates a non-white, non-black, not-hispanic person (=1).
# dep:        the number of dependents of each claimant;
# q1-q6:      six dummy variables indicating the quarter of experiment  during which each claimant enrolled.
# recall:     takes the value of 1 if the claimant answered ``yes'' when was asked if he/she had any expectation to be recalled
# agelt35:    takes the value of 1 if the claimant's age is less  than 35 and 0 otherwise.
# agegt54:    takes the value of 1 if the claimant's age is more than 54 and 0 otherwise.
# durable:    it takes the value of 1 if the occupation  of the claimant was in the sector of durable manufacturing and 0 otherwise.
# nondurable: it takes the value of 1 if the occupation of the claimant was in the sector of nondurable manufacturing and 0 otherwise.
# lusd:       it takes the value of 1 if the claimant filed  in Coatesville, Reading, or Lancaster and 0 otherwise.
#             These three sites were considered to be located in areas characterized by low unemployment rate and short duration of unemployment.
# husd:       it takes the value of 1 if the claimant filed in Lewistown, Pittston, or Scranton and 0 otherwise.
#             These three sites were considered to be located in areas characterized by high unemployment rate and short duration of unemployment.
# muld:       it takes the value of 1 if the claimant filed in Philadelphia-North, Philadelphia-Uptown, McKeesport, Erie, or Butler and 0 otherwise.
#             These three sites were considered to be located in areas characterized by moderate unemployment rate and long duration of unemployment."
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

################ Loading functions and Data ########################


rm(list = ls())  # Clear everything out so we're starting clean
source("ML_Functions.R")  
source("Moment_Functions.R")  
options(warn=-1)
set.seed(1210);
cl <- makeCluster(12, outfile="")

Penn<- as.data.frame(read.table("penn_jae.dat", header=T ));

########################### Sample Construction ######################

index       <- (Penn$tg==0) | (Penn$tg==4)
data        <- Penn[index,]
data$tg[(data$tg==4)] <- 1
data$dep  <- as.factor(data$dep)
data$inuidur1 <- log(data$inuidur1)

################################ Inputs ##############################

# Outcome Variable
y      <- "inuidur1";

# Treatment Indicator
d      <- "tg";  

# Controls
x      <- "female+black+othrace+ dep+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd"         # use this for tree-based methods like forests and boosted trees
xl     <- "(female+black+othrace+dep+q2+q3+q4+q5+q6+agelt35+agegt54+durable+lusd+husd)^2";     # use this for rlasso etc.

# Method names: Boosting, Nnet, RLasso, PostRLasso, Forest, Trees, Ridge, Lasso, Elnet, Ensemble

Boosting     <- list(bag.fraction = .5, train.fraction = 1.0, interaction.depth=2, n.trees=1000, shrinkage=.01, n.cores=1, cv.folds=5, verbose = FALSE, clas_dist= 'adaboost', reg_dist='gaussian')
Forest       <- list(clas_nodesize=1, reg_nodesize=5, ntree=1000, na.action=na.omit, replace=TRUE)
RLasso       <- list(penalty = list(homoscedastic = FALSE, X.dependent.lambda =FALSE, lambda.start = NULL, c = 1.1), intercept = TRUE)
Nnet         <- list(size=2,  maxit=1000, decay=0.02, MaxNWts=10000,  trace=FALSE)
Trees        <- list(reg_method="anova", clas_method="class")

arguments    <- list(Boosting=Boosting, Forest=Forest, RLasso=RLasso, Nnet=Nnet, Trees=Trees)

ensemble     <- list(methods=c("RLasso", "Boosting", "Forest", "Nnet"))              # methods for the ensemble estimation
methods      <- c("RLasso","Trees", "Forest", "Boosting", "Nnet", "Ensemble")        # ML methods that are used in estimation
split        <- 100                                                                  # number of splits


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
  
  dml <- DoubleML(data=data, y=y, d=d, z=NULL, xx=x, xL=xl, methods=methods, DML="DML2", nfold=2, est="plinear", arguments=arguments, ensemble=ensemble, silent=FALSE, trim=c(0.01,0.99)) 
  
  data.frame(t(dml[1,]), t(dml[2,]))
  
}

################################ Compute Output Table ########################################

result           <- matrix(0,3, length(methods)+1)
colnames(result) <- cbind(t(methods), "best")
rownames(result) <- cbind("Median ATE", "se(median)",  "se")

result[1,]        <- colQuantiles(r[,1:(length(methods)+1)], probs=0.5)
result[2,]        <- colQuantiles(sqrt(r[,(length(methods)+2):ncol(r)]^2+(r[,1:(length(methods)+1)] - colQuantiles(r[,1:(length(methods)+1)], probs=0.5))^2), probs=0.5)
result[3,]        <- colQuantiles(r[,(length(methods)+2):ncol(r)], probs=0.5)

result_table <- round(result, digits = 3)

for(i in 1:ncol(result_table)){
  for(j in seq(2,nrow(result_table),3)){
    
    result_table[j,i] <- paste("(", result_table[j,i], ")", sep="")
    
  }
  for(j in seq(3,nrow(result_table),3)){
    
    result_table[j,i] <- paste("(", result_table[j,i], ")", sep="")
    
  }
}

print(xtable(result_table, digits=3))

