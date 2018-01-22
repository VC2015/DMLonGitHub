
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

sim <- 500

check <- matrix(0, 2, sim)

for(i in 1:sim){

#  rm(list = ls())  # Clear everything out so we're starting clean
  source("ML_Functions.R")  
  source("Moment_Functions.R")  
  options(warn=-1)
  
  z<- as.numeric(rnorm (500)>0); x1 <- rnorm(500); x2 <- rnorm(500); x3 <- rnorm(500)
  error2 = rnorm(500)
  d <- as.numeric((2*z + x1 - 2*rnorm(500)>0))  # d endogenous covariate; z is exogenous
  y <- d + x3+ error2    # y outcome
  
  data <- data.frame(cbind(y,d,z, x1,x2,x3))
  
  y="y"
  d="d"
  z="z"; 
  methods=c("RLasso","Forest")
  
  dml <- DoubleML(data=data, y="y", d="d", z="z", xx="x1+x2+x3", xL="(x1+x2+x3)^2", methods=methods, DML="DML1", nfold=2, est="IV", arguments=NULL, ensemble=FALSE, silent=FALSE, trim=c(0.01, 0.99)) 
  
  check[,i] <- sapply(seq(1:2), function(x) ((dml[1,x] + 1.96*dml[2,x]) > 1 & (dml[1,x] - 1.96*dml[2,x]) < 1))

}

rowMeans(check)


lm.fit.ry              <- lm("y ~ d-1", data=data);
ate                    <- lm.fit.ry$coef;
HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');


lm.fit.ry              <- tsls(y=data$y, d=data$d, x=NULL, z=data$z, intercept = FALSE)
summary(lm.fit.ry)
