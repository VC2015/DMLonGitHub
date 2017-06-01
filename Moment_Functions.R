#########################################################################################################
#  Program:     Functions for estimating moments for using Machine Learning Methods.                    #
#  References: "Double/Debiased Machine Learning of Treatment and Causal Parameters",  AER P&P 2017     #
#              "Double Machine Learning for Treatment and Causal Parameters",  Arxiv 2016               #
#  by V.Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, W. Newey                         #           
#########################################################################################################

source("ML_Functions.R") 

DoubleML <- function(data, y, d,z, xx, xL, methods, DML, nfold, est, arguments, ensemble, silent=FALSE, trim){
  
  K         <- nfold
  TE        <- matrix(0,1,(length(methods)+1))
  STE       <- matrix(0,1,(length(methods)+1))
  result    <- matrix(0,2,(length(methods)+1))
  result2   <- matrix(0,2,(length(methods)+1))
  MSE1      <- matrix(0,length(methods)+1,K)
  MSE2      <- matrix(0,length(methods)+1,K)
  MSE3      <- matrix(0,length(methods)+1,K)
  MSE4      <- matrix(0,length(methods)+1,K)
  MSE5      <- matrix(0,length(methods)+1,K)
  cond.comp <- matrix(list(),length(methods),K)
  
  dpool     <- vector("list", (length(methods)+1))
  ypool     <- vector("list", (length(methods)+1))
  zopool    <- vector("list", (length(methods)+1))
  zpool     <- vector("list", (length(methods)+1))
  z1pool    <- vector("list", (length(methods)+1))
  z0pool    <- vector("list", (length(methods)+1))
  dz1pool   <- vector("list", (length(methods)+1))
  dz0pool   <- vector("list", (length(methods)+1))
  
  
  binary    <- as.numeric(checkBinary(data[,d]))
  
  flag      <- 0
  
  if(est=="LATE"){
    
    binary.z <- as.numeric(checkBinary(data[,z])) 
    if(!(binary.z==1)){
      print("instrument is not binary")
      stop()
    } 
    
    if(sum(!(data[data[,z]==0,d]==0))==0){
      
      flag <- 1
      
    }
  }
  
  split     <- runif(nrow(data))
  cvgroup   <- as.numeric(cut(split,quantile(split,probs = seq(0, 1, 1/K)),include.lowest = TRUE))  
  
  for(k in 1:length(methods)){   
    
    if(silent==FALSE){
      cat(methods[k],'\n')
    }
    
    if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==methods[k])){
      x=xL
    } else {
      x=xx
    }
    
    for(j in 1:K){   
      
      if(silent==FALSE){
        cat('  fold',j,'\n')
      }
      
      ii  <- cvgroup == j
      nii <- cvgroup != j
      
      if(K==1){
        
        ii  <- cvgroup == j
        nii <- cvgroup == j
        
      }
      
      datause <- as.data.frame(data[nii,])
      dataout <- as.data.frame(data[ii,]) 
      
      
      if(est=="LATE" && (length(methods)>0)){
        
        if(methods[k]=="Ensemble") { cond.comp[[k,j]] <- ensembleF(datause=datause, dataout=dataout, y=y, d=d, x=x, z=z, method=methods[k], plinear=3, xL=xL, binary=binary, flag=flag, arguments=arguments, ensemble=ensemble)}
        else{                       cond.comp[[k,j]] <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, x=x, z=z, method=methods[k], plinear=3, xL=xL, binary=binary, flag=flag, arguments=arguments);  }
        
        MSE1[k,j]               <- cond.comp[[k,j]]$err.yz0
        MSE2[k,j]               <- cond.comp[[k,j]]$err.yz1
        if(flag==1){ MSE3[k,j]  <- 0}
        else{  MSE3[k,j]        <- cond.comp[[k,j]]$err.dz0 }
        MSE4[k,j]               <- cond.comp[[k,j]]$err.dz1
        MSE5[k,j]               <- cond.comp[[k,j]]$err.z2
        
        drop                   <- which(cond.comp[[k,j]]$mz2_x>trim[1] & cond.comp[[k,j]]$mz2_x<trim[2])      
        mz_x                   <- cond.comp[[k,j]]$mz2_x[drop]
        my_z1x                 <- cond.comp[[k,j]]$my_z1x[drop]
        my_z0x                 <- cond.comp[[k,j]]$my_z0x[drop]
        md_z1x                 <- cond.comp[[k,j]]$md_z1x[drop]
        if(flag==1){ md_z0x    <- matrix(0,1,length(my_z0x))}
        else{  md_z0x          <- cond.comp[[k,j]]$md_z0x[drop] }
        yout                   <- dataout[drop,y]
        dout                   <- dataout[drop,d]
        zout                   <- dataout[drop,z]
        
        
        TE[1,k]                <- LATE(yout, dout, zout, my_z1x, my_z0x, mz_x, md_z1x, md_z0x)/K + TE[1,k];
        STE[1,k]               <- (1/(K^2))*((SE.LATE(yout, dout, zout, my_z1x, my_z0x, mz_x, md_z1x, md_z0x))^2) + STE[1,k];
        
        ypool[[k]]             <- c(ypool[[k]], yout)
        dpool[[k]]             <- c(dpool[[k]], dout)
        zopool[[k]]            <- c(zopool[[k]], zout)
        zpool[[k]]             <- c(zpool[[k]], mz_x)
        z1pool[[k]]            <- c(z1pool[[k]], my_z1x)
        z0pool[[k]]            <- c(z0pool[[k]], my_z0x)
        dz1pool[[k]]           <- c(dz1pool[[k]], md_z1x)
        dz0pool[[k]]           <- c(dz0pool[[k]], md_z0x)
        
        MSE1[(length(methods)+1),j] <- error(mean(datause[datause[,z]==0,y], na.rm = TRUE), dataout[!is.na(dataout[dataout[,z]==0,y]),y])$err
        MSE2[(length(methods)+1),j] <- error(mean(datause[datause[,z]==1,y], na.rm = TRUE), dataout[!is.na(dataout[dataout[,z]==1,y]),y])$err
        if(flag==1){ MSE3[(length(methods)+1),j]=0    }
        else{  MSE3[(length(methods)+1),j] <- error(mean(datause[datause[,z]==0,d], na.rm = TRUE), dataout[!is.na(dataout[dataout[,z]==0,d]),d])$err }
        MSE4[(length(methods)+1),j] <- error(mean(datause[datause[,z]==1,d], na.rm = TRUE), dataout[!is.na(dataout[dataout[,z]==1,d]),d])$err
        MSE5[(length(methods)+1),j] <- error(mean(datause[,z], na.rm = TRUE), dataout[!is.na(dataout[,z]),z])$err
        
      }
      
      
      
      if(est=="interactive" && (length(methods)>0)){
        
        if(methods[k]=="Ensemble") { cond.comp[[k,j]] <- ensembleF(datause=datause, dataout=dataout, y=y, d=d, x=x, method=methods[k], plinear=0, xL=xL, binary=binary, arguments=arguments, ensemble=ensemble)}
        else{                       cond.comp[[k,j]] <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, x=x, method=methods[k], plinear=0, xL=xL, binary=binary, arguments=arguments);  }
        
        MSE1[k,j]               <- cond.comp[[k,j]]$err.yz0
        MSE2[k,j]               <- cond.comp[[k,j]]$err.yz1
        MSE3[k,j]               <- cond.comp[[k,j]]$err.z
        
        drop                   <- which(cond.comp[[k,j]]$mz_x>trim[1] & cond.comp[[k,j]]$mz_x<trim[2])      
        mz_x                   <- cond.comp[[k,j]]$mz_x[drop]
        my_z1x                 <- cond.comp[[k,j]]$my_z1x[drop]
        my_z0x                 <- cond.comp[[k,j]]$my_z0x[drop]
        yout                   <- dataout[drop,y]
        dout                   <- dataout[drop,d]
        
        TE[1,k]                <- ATE(yout, dout, my_z1x, my_z0x, mz_x)/K + TE[1,k];
        STE[1,k]               <- (1/(K^2))*((SE.ATE(yout, dout, my_z1x, my_z0x, mz_x))^2) + STE[1,k];
        
        ypool[[k]]             <- c(ypool[[k]], yout)
        dpool[[k]]             <- c(dpool[[k]], dout)
        zpool[[k]]             <- c(zpool[[k]], mz_x)
        z1pool[[k]]            <- c(z1pool[[k]], my_z1x)
        z0pool[[k]]            <- c(z0pool[[k]], my_z0x)
        
        
        MSE1[(length(methods)+1),j] <- error(mean(datause[datause[,d]==0,y], na.rm = TRUE), dataout[!is.na(dataout[dataout[,d]==0,y]),y])$err
        MSE2[(length(methods)+1),j] <- error(mean(datause[datause[,d]==1,y], na.rm = TRUE), dataout[!is.na(dataout[dataout[,d]==1,y]),y])$err
        MSE3[(length(methods)+1),j] <- error(mean(datause[,d], na.rm = TRUE), dataout[!is.na(dataout[,d]),d])$err
        
      }
      
      if(est=="plinear" && (length(methods)>0)){
        
        
        if(methods[k]=="Ensemble") { cond.comp[[k,j]] <- ensembleF(datause=datause, dataout=dataout, y=y, d=d, x=x, method=methods[k], plinear=1, xL=xL, binary=binary, arguments=arguments, ensemble=ensemble)}
        else{                        cond.comp[[k,j]] <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, x=x, method=methods[k], plinear=1, xL=xL, binary=binary, arguments=arguments)}
        
        
        MSE1[k,j]              <- cond.comp[[k,j]]$err.y
        MSE2[k,j]              <- cond.comp[[k,j]]$err.z
        
        lm.fit.ry              <- lm(as.matrix(cond.comp[[k,j]]$ry) ~ as.matrix(cond.comp[[k,j]]$rz)-1);
        ate                    <- lm.fit.ry$coef;
        HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
        STE[1,k]               <- (1/(K^2))*(diag(HCV.coefs)) +  STE[1,k] 
        TE[1,k]                <- ate/K + TE[1,k] ;
        
        ypool[[k]]             <- c(ypool[[k]], cond.comp[[k,j]]$ry)
        zpool[[k]]             <- c(zpool[[k]], cond.comp[[k,j]]$rz)
        
        
        MSE1[(length(methods)+1),j] <- error(rep(mean(datause[,y], na.rm = TRUE), length(dataout[!is.na(dataout[,y]),y])), dataout[!is.na(dataout[,y]),y])$err
        MSE2[(length(methods)+1),j] <- error(rep(mean(datause[,d], na.rm = TRUE), length(dataout[!is.na(dataout[,d]),d])), dataout[!is.na(dataout[,d]),d])$err
        
      }
      
      if(est=="IV" && (length(methods)>0)){
        
        if(methods[k]=="Ensemble") { 
          cond.comp[[k,j]] <- ensembleF(datause=datause, dataout=dataout, y=y, d=d, z=z, x=x, method=methods[k], plinear=2, xL=xL, binary=binary, arguments=arguments, ensemble=ensemble)
        }
        else{       
          cond.comp[[k,j]] <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, z=z, x=x, method=methods[k], plinear=2, xL=xL, binary=binary, arguments=arguments)
        }
        
        MSE1[k,j]              <- cond.comp[[k,j]]$err.y
        MSE2[k,j]              <- cond.comp[[k,j]]$err.z
        MSE3[k,j]              <- cond.comp[[k,j]]$err.z2
        
        lm.fit.ry              <- tsls(y=cond.comp[[k,j]]$ry,d=cond.comp[[k,j]]$rz, x=NULL, z=cond.comp[[k,j]]$rz2, intercept = FALSE)
        ate                    <- lm.fit.ry$coef[1];
        HCV.coefs              <- sqrt(lm.fit.ry$vcov[1])
        
        
        STE[1,k]               <- (1/(K^2))*((HCV.coefs)) +  STE[1,k] 
        TE[1,k]                <- ate/K + TE[1,k] ;
        
        ypool[[k]]             <- c(ypool[[k]], cond.comp[[k,j]]$ry)
        zpool[[k]]             <- c(zpool[[k]], cond.comp[[k,j]]$rz2)
        dpool[[k]]             <- c(dpool[[k]], cond.comp[[k,j]]$rz)
        
        
        MSE1[(length(methods)+1),j] <- error(mean(datause[,y], na.rm = TRUE), dataout[!is.na(dataout[,y]),y])$err
        MSE2[(length(methods)+1),j] <- error(mean(datause[,d], na.rm = TRUE), dataout[!is.na(dataout[,d]),d])$err
        MSE3[(length(methods)+1),j] <- error(mean(datause[,z], na.rm = TRUE), dataout[!is.na(dataout[,z]),z])$err
        
      }
    }  
  }
  
  if(length(methods)>1){
    
    if(est=="LATE"){
      
      min1 <- which.min(rowMeans(MSE1[1:length(methods),]))
      min2 <- which.min(rowMeans(MSE2[1:length(methods),]))
      min3 <- which.min(rowMeans(MSE3[1:length(methods),]))
      min4 <- which.min(rowMeans(MSE4[1:length(methods),]))
      min5 <- which.min(rowMeans(MSE5[1:length(methods),]))
      
      if(silent==FALSE){
        cat('  best methods for E[Y|X, D=0]:',methods[min1],'\n')
        cat('  best methods for E[Y|X, D=1]:',methods[min2],'\n')
        cat('  best methods for E[D|X, Z=0]:',methods[min3],'\n')
        cat('  best methods for E[D|X, Z=1]:',methods[min4],'\n')
        cat('  best methods for E[Z|X]:',methods[min5],'\n')
      }
    }
    
    
    if(est=="interactive"){
      
      min1 <- which.min(rowMeans(MSE1[1:length(methods),]))
      min2 <- which.min(rowMeans(MSE2[1:length(methods),]))
      min3 <- which.min(rowMeans(MSE3[1:length(methods),]))
      
      if(silent==FALSE){
        cat('  best methods for E[Y|X, D=0]:',methods[min1],'\n')
        cat('  best methods for E[Y|X, D=1]:',methods[min2],'\n')
        cat('  best methods for E[D|X]:',methods[min3],'\n')
      }
    }
    
    if(est=="plinear"){
      
      min1 <- which.min(rowMeans(MSE1[1:length(methods),]))
      min2 <- which.min(rowMeans(MSE2[1:length(methods),]))
      
      if(silent==FALSE){   
        cat('  best methods for E[Y|X]:',methods[min1],'\n')
        cat('  best methods for E[D|X]:',methods[min2],'\n')
      }    
    }
    
    if(est=="IV"){
      
      min1 <- which.min(rowMeans(MSE1[1:length(methods),]))
      min2 <- which.min(rowMeans(MSE2[1:length(methods),]))
      min3 <- which.min(rowMeans(MSE3[1:length(methods),]))
      
      if(silent==FALSE){   
        cat('  best methods for E[Y|X]:',methods[min1],'\n')
        cat('  best methods for E[D|X]:',methods[min2],'\n')
        cat('  best methods for E[Z|X]:',methods[min3],'\n')
      }    
    }
    
    
    for(j in 1:K){  
      
      ii = cvgroup == j
      nii = cvgroup != j
      
      datause = as.data.frame(data[nii,])
      dataout = as.data.frame(data[ii,])  
      
      
      if(est=="LATE"){
        
        drop                   <- which(cond.comp[[min5,j]]$mz2_x>trim[1] & cond.comp[[min5,j]]$mz2_x<trim[2])      
        mz_x                   <- cond.comp[[min1,j]]$mz2_x[drop]
        my_z1x                 <- cond.comp[[min2,j]]$my_z1x[drop]
        my_z0x                 <- cond.comp[[min3,j]]$my_z0x[drop]
        md_z1x                 <- cond.comp[[min4,j]]$md_z1x[drop]
        if(flag==1){ md_z0x    <- matrix(0,1,length(my_z0x))}
        else{  md_z0x          <- cond.comp[[min5,j]]$md_z0x[drop] }
        
        yout                   <- dataout[drop,y]
        dout                   <- dataout[drop,d]
        zout                   <- dataout[drop,z]
        
        TE[1,(k+1)]            <- LATE(yout, dout, zout, my_z1x, my_z0x, mz_x, md_z1x, md_z0x)/K + TE[1,(k+1)];
        STE[1,(k+1)]           <- (1/(K^2))*((SE.LATE(yout, dout, zout, my_z1x, my_z0x, mz_x, md_z1x, md_z0x))^2) + STE[1,(k+1)];
        
        ypool[[k+1]]             <- c(ypool[[k+1]], yout)
        dpool[[k+1]]             <- c(dpool[[k+1]], dout)
        zopool[[k+1]]            <- c(zopool[[k+1]], zout)
        zpool[[k+1]]             <- c(zpool[[k+1]], mz_x)
        z1pool[[k+1]]            <- c(z1pool[[k+1]], my_z1x)
        z0pool[[k+1]]            <- c(z0pool[[k+1]], my_z0x)
        dz1pool[[k+1]]           <- c(dz1pool[[k+1]], md_z1x)
        dz0pool[[k+1]]           <- c(dz0pool[[k+1]], md_z0x)
        
      }
      
      
      if(est=="interactive"){
        
        drop                   <- which(cond.comp[[min3,j]]$mz_x>trim[1] & cond.comp[[min3,j]]$mz_x<trim[2])      
        mz_x                   <- cond.comp[[min1,j]]$mz_x[drop]
        my_z1x                 <- cond.comp[[min2,j]]$my_z1x[drop]
        my_z0x                 <- cond.comp[[min3,j]]$my_z0x[drop]
        yout                   <- dataout[drop,y]
        dout                   <- dataout[drop,d]
        
        TE[1,(k+1)]            <- ATE(yout, dout, my_z1x, my_z0x, mz_x)/K + TE[1,(k+1)];
        STE[1,(k+1)]           <- (1/(K^2))*((SE.ATE(yout, dout, my_z1x, my_z0x, mz_x))^2) + STE[1,(k+1)];
        
        ypool[[k+1]]             <- c(ypool[[k+1]], yout)
        dpool[[k+1]]             <- c(dpool[[k+1]], dout)
        zpool[[k+1]]             <- c(zpool[[k+1]], mz_x)
        z1pool[[k+1]]            <- c(z1pool[[k+1]], my_z1x)
        z0pool[[k+1]]            <- c(z0pool[[k+1]], my_z0x)
        
      }
      
      if(est=="plinear"){
        
        lm.fit.ry              <- lm(as.matrix(cond.comp[[min1,j]]$ry) ~ as.matrix(cond.comp[[min2,j]]$rz)-1);
        ate                    <- lm.fit.ry$coef;
        HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
        STE[1,(k+1)]           <- (1/(K^2))*(diag(HCV.coefs)) +  STE[1,(k+1)] 
        TE[1,(k+1)]            <- ate/K + TE[1,(k+1)] ;
        
        ypool[[k+1]]             <- c(ypool[[k+1]], cond.comp[[min1,j]]$ry)
        zpool[[k+1]]             <- c(zpool[[k+1]], cond.comp[[min2,j]]$rz)
        
        
      }
      
      if(est=="IV"){
        
        lm.fit.ry              <- tsls(y=cond.comp[[min1,j]]$ry, d=cond.comp[[min2,j]]$rz, x=NULL, z=cond.comp[[min3,j]]$rz2, intercept = FALSE)
        ate                    <- lm.fit.ry$coef[1];
        HCV.coefs              <- sqrt(lm.fit.ry$vcov[1])
        
        
        STE[1,(k+1)]           <- (1/(K^2))*((HCV.coefs)) +  STE[1,(k+1)] 
        TE[1,(k+1)]            <- ate/K + TE[1,(k+1)] ;
        
        ypool[[k+1]]             <- c(ypool[[k+1]], cond.comp[[k,j]]$ry)
        zpool[[k+1]]             <- c(zpool[[k+1]], cond.comp[[k,j]]$rz2)
        dpool[[k+1]]             <- c(dpool[[k+1]], cond.comp[[k,j]]$rz)
        
      }
    }
  }
  
  
  TE_pool        <- matrix(0,1,(length(methods)+1))
  STE_pool       <- matrix(0,1,(length(methods)+1))
  
  for(k in 1:(length(methods)+1)){ 
    
    if(est=="LATE"){
      
      TE_pool[1,(k)]         <- LATE(ypool[[k]], dpool[[k]], zopool[[k]],  z1pool[[k]], z0pool[[k]], zpool[[k]], dz1pool[[k]], dz0pool[[k]])
      STE_pool[1,(k)]        <- ((SE.LATE(ypool[[k]], dpool[[k]], zopool[[k]],  z1pool[[k]], z0pool[[k]], zpool[[k]], dz1pool[[k]], dz0pool[[k]]))^2) 
      
    }
    
    if(est=="interactive"){
      
      TE_pool[1,(k)]         <- ATE(ypool[[k]], dpool[[k]], z1pool[[k]], z0pool[[k]], zpool[[k]])
      STE_pool[1,(k)]        <- ((SE.ATE(ypool[[k]], dpool[[k]], z1pool[[k]], z0pool[[k]], zpool[[k]]))^2)
      
    }
    
    if(est=="plinear"){
      
      lm.fit.ry              <- lm(as.matrix(ypool[[k]]) ~ as.matrix(zpool[[k]])-1);
      ate                    <- lm.fit.ry$coef;
      HCV.coefs              <- vcovHC(lm.fit.ry, type = 'HC');
      STE_pool[1,k]          <- (diag(HCV.coefs))
      TE_pool[1,k]           <- ate
      
    }
    
    if(est=="IV"){
      
      lm.fit.ry              <- tsls(y=ypool[[k]],d=dpool[[k]], x=NULL, z=zpool[[k]], intercept = FALSE)
      ate                    <- lm.fit.ry$coef[1];
      HCV.coefs              <- sqrt(lm.fit.ry$vcov[1])
      
      STE_pool[1,k]          <- HCV.coefs
      TE_pool[1,k]           <- ate
    }
  }
  
  if(length(methods)==1){
    
    TE_pool[1,(length(methods)+1)]  <- TE_pool[1,(length(methods))]
    STE_pool[1,(length(methods)+1)] <- STE_pool[1,(length(methods))]
    
  }
  
  colnames(result)   <- c(methods, "best") 
  colnames(result2)  <- c(methods, "best") 
  rownames(MSE1)     <- c(methods, "best") 
  rownames(MSE2)     <- c(methods, "best") 
  rownames(MSE3)     <- c(methods, "best") 
  rownames(result)   <- c("ATE", "se")
  rownames(result2)  <- c("ATE", "se")
  
  if(DML=="DML1"){
    result[1,]         <- colMeans(TE)
    result[2,]         <- sqrt((STE))
  }
  
  if(DML=="DML2"){
    result[1,]         <- colMeans(TE_pool)
    result[2,]         <- sqrt((STE_pool))
  }
  
  
  if(est=="plinear"){   
    table <- rbind(result, rowMeans(MSE1), rowMeans(MSE2)) 
    rownames(table)[3:4]   <- c("MSE[Y|X]", "MSE[D|X]") 
  }
  
  if(est=="IV"){   
    table <- rbind(result, rowMeans(MSE1), rowMeans(MSE2) , rowMeans(MSE3)) 
    rownames(table)[3:5]   <- c("MSE[Y|X]", "MSE[D|X]", "MSE[Z|X]") 
  }
  
  if(est=="interactive"){    
    table <- rbind(result, rowMeans(MSE1), rowMeans(MSE2) , rowMeans(MSE3))   
    rownames(table)[3:5]   <- c("MSE[Y|X, D=0]", "MSE[Y|X, D=1]", "MSE[D|X]")
  }  
  
  if(est=="LATE"){    
    table <- rbind(result, rowMeans(MSE1), rowMeans(MSE2) , rowMeans(MSE3),rowMeans(MSE4) , rowMeans(MSE5))    
    rownames(table)[3:7]   <- c("MSE[Y|X, Z=0]", "MSE[Y|X, Z=1]", "MSE[D|X,Z=0]", "MSE[D|X,Z=1]" ,"MSE[Z|X]")
  }  
  
  colnames(table)[length(methods)+1] = "best"
  return(table)
}  

cond_comp <- function(datause, dataout, y, d, z=NULL, x, method, plinear,xL, binary,flag=0, arguments){
  
  form_y   <- y
  form_d   <- d
  form_z   <- z
  form_x   <- x
  form_xL  <- xL
  ind_u    <- which(datause[,d]==1)
  ind_o    <- which(dataout[,d]==1)
  if(plinear==3){
    ind_u    <- which(datause[,z]==1)
    ind_o    <- which(dataout[,z]==1)
  }
  err.yz1  <- NULL
  err.yz0  <- NULL
  my_z1x   <- NULL
  my_z0x   <- NULL
  fit.yz1  <- NULL
  fit.yz0  <- NULL
  fit.z2   <- NULL
  rz2      <- NULL
  mz2_x    <- NULL
  err.z2   <- NULL
  fit.dz1  <- NULL
  err.dz1  <- NULL
  md_z1x   <- NULL
  fit.dz0  <- NULL
  err.dz0  <- NULL
  md_z0x   <- NULL
  
  
  ########################## Boosted  Trees ###################################################;
  
  if(method=="Boosting")
  {
    
    option <- arguments[[method]]
    arg    <- option
    arg[which(names(arg) %in% c("clas_dist","reg_dist"))] <-  NULL
    
    if(plinear==3){
      
      fit.yz1        <- boost(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, distribution=option[['reg_dist']], option=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, n.trees=fit.yz1$best, dataout, type="response") 
      
      fit.dz1        <- boost(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_d, distribution=option[['clas_dist']], option=arg)
      err.dz1        <- error(fit.dz1$yhatout, dataout[ind_o,y])$err
      md_z1x         <- predict(fit.dz1$model, n.trees=fit.dz1$best, dataout, type="response") 
      
      fit.yz0        <- boost(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, distribution=option[['reg_dist']], option=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,d])$err
      my_z0x         <- predict(fit.yz0$model, n.trees=fit.yz0$best, dataout, type="response") 
      
      if(flag==0){
        fit.dz0        <- boost(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_d, distribution=option[['clas_dist']], option=arg)
        err.dz0        <- error(fit.dz0$yhatout, dataout[-ind_o,d])$err
        md_z0x         <- predict(fit.dz0$model, n.trees=fit.dz0$best, dataout, type="response") 
      }
    }
    
    if(plinear==0){
      
      fit.yz1        <- boost(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, distribution=option[['reg_dist']], option=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, n.trees=fit.yz1$best, dataout, type="response") 
      
      fit.yz0        <- boost(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, distribution=option[['reg_dist']], option=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model, n.trees=fit.yz0$best, dataout, type="response") 
      
      
    }
    
    if(binary==1){
      fit.z          <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, distribution=option[['clas_dist']], option=arg)
      mis.z          <- error(fit.z$yhatout, dataout[,d])$mis
    }
    
    
    if(binary==0){
      fit.z          <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, distribution=option[['reg_dist']], option=arg)
      mis.z          <- NA
    }
    
    fit.y            <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, distribution=option[['reg_dist']], option=arg)
    
    if(plinear==2 | plinear==3){
      fit.z2         <- boost(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_z, distribution=option[['reg_dist']], option=arg)
    }
    
  }  
  
  
  ########################## Neural Network(Nnet Package) ###################################################;   
  
  
  if(method=="Nnet"){
    
    option <- arguments[[method]]
    arg    <- option
    
    if(plinear==3){
      
      fit.yz1        <- nnetF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, arg=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      dataouts       <- dataout
      dataouts[,!fit.yz1$f] <- as.data.frame(scale(dataouts[,!fit.yz1$f], center = fit.yz1$min, scale = fit.yz1$max - fit.yz1$min))
      my_z1x         <- predict(fit.yz1$model, dataouts)*(fit.yz1$max[fit.yz1$k]-fit.yz1$min[fit.yz1$k])+fit.yz1$min[fit.yz1$k] 
      
      fit.yz0        <- nnetF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, arg=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      dataouts       <- dataout
      dataouts[,!fit.yz0$f] <- as.data.frame(scale(dataouts[,!fit.yz0$f], center = fit.yz0$min, scale = fit.yz0$max - fit.yz0$min))
      my_z0x         <- predict(fit.yz0$model, dataouts)*(fit.yz0$max[fit.yz0$k]-fit.yz0$min[fit.yz0$k])+fit.yz0$min[fit.yz0$k] 
      
      fit.dz1        <- nnetF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_d, arg=arg)
      err.dz1        <- error(fit.dz1$yhatout, dataout[ind_o,d])$err
      dataouts       <- dataout
      dataouts[,!fit.dz1$f] <- as.data.frame(scale(dataouts[,!fit.dz1$f], center = fit.dz1$min, scale = fit.dz1$max - fit.dz1$min))
      md_z1x         <- predict(fit.dz1$model, dataouts)*(fit.dz1$max[fit.dz1$k]-fit.dz1$min[fit.dz1$k])+fit.yz1$min[fit.dz1$k] 
      
      if(flag==0){
        fit.dz0        <- nnetF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_d, arg=arg)
        err.dz0        <- error(fit.dz0$yhatout, dataout[-ind_o,d])$err
        dataouts       <- dataout
        dataouts[,!fit.dz0$f] <- as.data.frame(scale(dataouts[,!fit.dz0$f], center = fit.dz0$min, scale = fit.dz0$max - fit.dz0$min))
        md_z0x         <- predict(fit.dz0$model, dataouts)*(fit.dz0$max[fit.dz0$k]-fit.dz0$min[fit.dz0$k])+fit.dz0$min[fit.dz0$k] 
      }
    }
    
    if(plinear==0){
      
      fit.yz1        <- nnetF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, arg=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      dataouts       <- dataout
      dataouts[,!fit.yz1$f] <- as.data.frame(scale(dataouts[,!fit.yz1$f], center = fit.yz1$min, scale = fit.yz1$max - fit.yz1$min))
      my_z1x         <- predict(fit.yz1$model, dataouts)*(fit.yz1$max[fit.yz1$k]-fit.yz1$min[fit.yz1$k])+fit.yz1$min[fit.yz1$k] 
      
      fit.yz0        <- nnetF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, arg=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      dataouts       <- dataout
      dataouts[,!fit.yz0$f] <- as.data.frame(scale(dataouts[,!fit.yz0$f], center = fit.yz0$min, scale = fit.yz0$max - fit.yz0$min))
      my_z0x         <- predict(fit.yz0$model, dataouts)*(fit.yz0$max[fit.yz0$k]-fit.yz0$min[fit.yz0$k])+fit.yz0$min[fit.yz0$k] 
    }
    
    if(binary==1){
      fit.z          <- nnetF(datause=datause, dataout=dataout, form_x=form_x, form_y=form_d, clas=TRUE, arg=arg)
      mis.z          <- error(fit.z$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit.z          <- nnetF(datause=datause, dataout=dataout, form_x=form_x, form_y=form_d, clas=FALSE, arg=arg)
      mis.z          <- NA
    }
    
    fit.y          <- nnetF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, arg=arg)
    
    if(plinear==2 | plinear==3){
      fit.z2           <- nnetF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_z, arg=arg)
    }
    
  } 
  
  ########################## Lasso and Post Lasso(Hdm Package) ###################################################;    
  
  if(method=="RLasso" || method=="PostRLasso"){
    
    post = FALSE
    if(method=="PostRLasso"){ post=TRUE }
    
    option    <- arguments[[method]]
    arg       <- option
    
    if(plinear==3){
      
      fit.yz1        <- rlassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, post, arg=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, newdata=formC(form_y, form_x, dataout)$x , type="response") 
      
      fit.yz0        <- rlassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, post, arg=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model, newdata=formC(form_y, form_x, dataout)$x, type="response")   
      
      fit.dz1        <- rlassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_d, post, arg=arg)
      err.dz1        <- error(fit.dz1$yhatout, dataout[ind_o,d])$err
      md_z1x         <- predict(fit.dz1$model, newdata=formC(form_y, form_x, dataout)$x , type="response") 
      
      if(flag==0){
        fit.dz0        <- rlassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_d, post, arg=arg)
        err.dz0        <- error(fit.dz0$yhatout, dataout[-ind_o,d])$err
        md_z0x         <- predict(fit.dz0$model, newdata=formC(form_y, form_x, dataout)$x, type="response")   
      }
    }
    
    if(plinear==0){
      
      fit.yz1        <- rlassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, post, arg=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, newdata=formC(form_y, form_x, dataout)$x , type="response") 
      
      fit.yz0        <- rlassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, post, arg=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model, newdata=formC(form_y, form_x, dataout)$x, type="response")   
      
    }
    
    if(binary==1){
      fit.z          <- rlassoF(datause=datause, dataout=dataout,  form_x, form_d, post, logit=TRUE, arg=arg)
      mis.z          <- error(fit.z$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit.z          <- rlassoF(datause=datause, dataout=dataout,  form_x, form_d, post, logit=FALSE, arg=arg)
      mis.z          <- NA
    }   
    
    fit.y          <- rlassoF(datause=datause, dataout=dataout,  form_x, form_y, post, arg=arg)
    
    if(plinear==2 | plinear==3){
      fit.z2         <- rlassoF(datause=datause, dataout=dataout,  form_x, form_z, post, arg=arg)
    }
    
  }    
  
  
  ########################## Lasso and Post Lasso(Glmnet) Package) ###################################################;    
  
  if(method=="Ridge" || method=="Lasso" || method=="Elnet"){
    
    if(method=="Ridge"){ alp=0 }
    if(method=="Lasso"){ alp=1 }
    if(method=="Elnet"){ alp=0.5 }
    
    option    <- arguments[[method]]
    arg       <- option
    arg[which(names(arg) %in% c("s"))] <-  NULL
    s         <- option[['s']]
    
    if(plinear==3){
      
      fit.yz1        <- lassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, alp=alp, arg=arg, s=s)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      fit.p          <- lm(as.formula(paste(form_y, "~", form_x)),  x = TRUE, y = TRUE, data=dataout);
      my_z1x         <- predict(fit.yz1$model, newx=fit.p$x[,-1] ) 
      
      fit.yz0        <- lassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, alp=alp, arg=arg, s=s)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      fit.p          <- lm(as.formula(paste(form_y, "~", form_x)),  x = TRUE, y = TRUE, data=dataout);
      my_z0x         <- predict(fit.yz0$model,  newx=fit.p$x[,-1])   
      
      fit.dz1        <- lassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_d, alp=alp, arg=arg, s=s)
      err.dz1        <- error(fit.yz1$yhatout, dataout[ind_o,d])$err
      fit.p          <- lm(as.formula(paste(form_y, "~", form_x)),  x = TRUE, y = TRUE, data=dataout);
      md_z1x         <- predict(fit.dz1$model, newx=fit.p$x[,-1] ) 
      
      if(flag==0){
        fit.dz0        <- lassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_d, alp=alp, arg=arg, s=s)
        err.dz0        <- error(fit.yz0$yhatout, dataout[-ind_o,d])$err
        fit.p          <- lm(as.formula(paste(form_y, "~", form_x)),  x = TRUE, y = TRUE, data=dataout);
        md_z0x         <- predict(fit.dz0$model,  newx=fit.p$x[,-1])   
      }
    }
    
    if(plinear==0){
      
      fit.yz1        <- lassoF(datause=datause[ind_u,], dataout=dataout[ind_o,],  form_x, form_y, alp=alp, arg=arg, s=s)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      fit.p          <- lm(as.formula(paste(form_y, "~", form_x)),  x = TRUE, y = TRUE, data=dataout);
      my_z1x         <- predict(fit.yz1$model, newx=fit.p$x[,-1] ) 
      
      fit.yz0        <- lassoF(datause=datause[-ind_u,], dataout=dataout[-ind_o,], form_x, form_y, alp=alp, arg=arg, s=s)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      fit.p          <- lm(as.formula(paste(form_y, "~", form_x)),  x = TRUE, y = TRUE, data=dataout);
      my_z0x         <- predict(fit.yz0$model,  newx=fit.p$x[,-1])   
      
    }
    
    if(binary==1){
      fit.z          <- lassoF(datause=datause, dataout=dataout,  form_x, form_d, logit=TRUE, alp=alp, arg=arg, s=s)
      mis.z          <- error(fit.z$yhatout, dataout[,d])$mis
    }
    
    if(binary==0){
      fit.z          <- lassoF(datause=datause, dataout=dataout,  form_x, form_d, logit=FALSE, alp=alp, arg=arg, s=s)
      mis.z          <- NA
    }   
    
    fit.y            <- lassoF(datause=datause, dataout=dataout,  form_x, form_y, alp=alp, arg=arg, s=s)
    
    if(plinear==2 | plinear==3){
      fit.z2         <- lassoF(datause=datause, dataout=dataout,  form_x, form_z, alp=alp, arg=arg, s=s)
    }
    
  }    
  
  ############# Random Forest ###################################################;
  
  if(method=="Forest" | method=="TForest"){
    
    tune = FALSE
    if(method=="TForest"){tune=TRUE}
    
    option    <- arguments[[method]]
    
    arg       <- option
    arg[which(names(arg) %in% c("clas_nodesize","reg_nodesize"))] <-  NULL
    
    if(plinear==3){
      
      fit.yz1        <- RF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, nodesize=option[["reg_nodesize"]], arg=arg, tune=tune)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, dataout, type="response") 
      
      fit.yz0        <- RF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, nodesize=option[["reg_nodesize"]], arg=arg, tune=tune)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model, dataout, type="response")
      
      fit.dz1        <- RF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_d, nodesize=option[["clas_nodesize"]], arg=arg, tune=tune)
      err.dz1        <- error(fit.dz1$yhatout, dataout[ind_o,d])$err
      md_z1x         <- predict(fit.dz1$model, dataout, type="response") 
      
      if(flag==0){
        fit.dz0        <- RF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_d, nodesize=option[["clas_nodesize"]], arg=arg, tune=tune)
        err.dz0        <- error(fit.dz0$yhatout, dataout[-ind_o,d])$err
        md_z0x         <- predict(fit.dz0$model, dataout, type="response")
      }
    }
    
    if(plinear==0){
      
      fit.yz1        <- RF(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, nodesize=option[["reg_nodesize"]], arg=arg, tune=tune)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, dataout, type="response") 
      
      fit.yz0        <- RF(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, nodesize=option[["reg_nodesize"]], arg=arg, tune=tune)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model, dataout, type="response")
      
    }
    
    if(binary==1){
      fit.z          <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, nodesize=option[["clas_nodesize"]], arg=arg, reg=TRUE, tune=tune)
      mis.z          <- error(as.numeric(fit.z$yhatout), dataout[,y])$mis
    }
    
    if(binary==0){
      fit.z          <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d,nodesize=option[["reg_nodesize"]], arg=arg, reg=TRUE, tune=tune)
      mis.z          <- NA
    }   
    
    fit.y           <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, nodesize=option[["reg_nodesize"]],  arg=arg, tune=tune)
    
    if(plinear==2 | plinear==3){
      fit.z2          <- RF(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_z, nodesize=option[["clas_nodesize"]],  arg=arg, tune=tune)
    }   
    
  }
  
  ########################## Regression Trees ###################################################;     
  
  if(method=="Trees"){
    
    option    <- arguments[[method]]
    arg       <- option
    arg[which(names(arg) %in% c("reg_method","clas_method"))] <-  NULL
    
    if(plinear==3){
      
      fit.yz1        <- tree(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, method=option[["reg_method"]], arg=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, dataout) 
      
      fit.yz0        <- tree(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, method=option[["reg_method"]], arg=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model,dataout)   
      
      fit.dz1        <- tree(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_d, method=option[["clas_method"]], arg=arg)
      err.dz1        <- error(fit.dz1$yhatout, dataout[ind_o,d])$err
      md_z1x         <- predict(fit.dz1$model, dataout) 
      
      if(flag==0){
        fit.dz0        <- tree(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_d, method=option[["clas_method"]], arg=arg)
        err.dz0        <- error(fit.dz0$yhatout, dataout[-ind_o,d])$err
        md_z0x         <- predict(fit.dz0$model,dataout)  
      }
      fit.z2         <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_z, method=option[["clas_method"]], arg=arg)
      
    }
    
    if(plinear==0){
      
      fit.yz1        <- tree(datause=datause[ind_u,], dataout=dataout[ind_o,], form_x=form_x,  form_y=form_y, method=option[["reg_method"]], arg=arg)
      err.yz1        <- error(fit.yz1$yhatout, dataout[ind_o,y])$err
      my_z1x         <- predict(fit.yz1$model, dataout) 
      
      fit.yz0        <- tree(datause=datause[-ind_u,], dataout=dataout[-ind_o,],  form_x=form_x, form_y=form_y, method=option[["reg_method"]], arg=arg)
      err.yz0        <- error(fit.yz0$yhatout, dataout[-ind_o,y])$err
      my_z0x         <- predict(fit.yz0$model,dataout)   
      
    }
    
    
    if(binary==1){
      
      fit.z          <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, method=option[["clas_method"]], arg=arg)
      mis.z          <- error(as.numeric(fit.z$yhatout), dataout[,y])$mis
    }
    
    if(binary==0){
      
      fit.z          <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_d, method=option[["reg_method"]], arg=arg)
      mis.z          <- NA
    }        
    
    fit.y           <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_y, method=option[["cont_method"]], arg=arg)
    
    if(plinear==2){
      fit.z2          <- tree(datause=datause, dataout=dataout,  form_x=form_x, form_y=form_z, method=option[["cont_method"]], arg=arg)
    }   
  }
  
  err.z          <- error(fit.z$yhatout, dataout[,d])$err
  mz_x           <- fit.z$yhatout       
  rz             <- fit.z$resout 
  err.z          <- error(fit.z$yhatout, dataout[,d])$err 
  
  ry             <- fit.y$resout
  my_x           <- fit.y$yhatout
  err.y          <- error(fit.y$yhatout, dataout[,y])$err
  
  if(plinear==2 | plinear==3){
    rz2            <- fit.z2$resout
    mz2_x          <- fit.z2$yhatout
    err.z2         <- error(fit.z2$yhatout, dataout[,z])$err
  }   
  
  return(list(fit.dz0 = fit.dz0, err.dz0=err.dz0, md_z0x=md_z0x, fit.dz1 = fit.dz1, err.dz1=err.dz1, md_z1x=md_z1x, rz2=rz2, mz2_x=mz2_x, err.z2=err.z2, my_z1x=my_z1x, mz_x= mz_x, my_z0x=my_z0x, my_x = my_x, err.z = err.z,  err.yz0= err.yz0,  err.yz1=err.yz1, mis.z=mis.z, ry=ry , rz=rz, err.y=err.y,  fit.y=fit.y, fit.yz1out= fit.yz1$yhatout,  fit.yz0out= fit.yz0$yhatout));
  
}  

ensembleF <- function(datause, dataout, y, d, z, xx, method, plinear, xL, binary, flag=flag, arguments, ensemble){
  
  K         <- 2
  k         <- length(ensemble[['methods']])
  fits      <- vector("list", k)
  method    <- ensemble[['methods']]
  ind_u     <- which(datause[,d]==1)
  ind_o     <- which(dataout[,d]==1)
  if(plinear==3){
    ind_u     <- which(datause[,z]==1)
    ind_o     <- which(dataout[,z]==1)
  }
  err.z2    <- NULL
  rz2       <- NULL
  
  split     <- runif(nrow(datause))
  cvgroup   <- as.numeric(cut(split,quantile(split,probs = seq(0, 1, 1/K)),include.lowest = TRUE))  
  
  if(k<4)  {  lst <- lapply(numeric(k), function(x) as.vector(seq(0,1,0.01))) }
  if(k==4) {  lst <- lapply(numeric(k), function(x) as.vector(seq(0,1,0.02))) }
  if(k==5) {  lst <- lapply(numeric(k), function(x) as.vector(seq(0,1,0.04))) }
  if(k==6) {  lst <- lapply(numeric(k), function(x) as.vector(seq(0,1,0.1)))  }
  
  gr      <- as.matrix(expand.grid(lst))
  weight  <- gr[rowSums(gr)==1,]
  
  
  if(plinear==1 | plinear==2){
    
    errorM  <- array(0,dim=c(nrow(weight),5,3))
    pred2   <- array(0,dim=c(nrow(dataout),3,k))  
    
    for(j in 1:K){
      
      ii   <- cvgroup == j
      nii  <- cvgroup != j
      
      datause1 <- as.data.frame(datause[nii,])
      datause2 <- as.data.frame(datause[ii,])  
      pred1    <- array(0,dim=c(nrow(datause2),3,k))  
      
      for(i in 1:k){
        
        if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==method[i])){
          x=xL
        } else {
          x=xx
        }
        
        if(plinear==2){
          fits[[i]]   <- cond_comp(datause=datause1, dataout=datause2, y=y, d=d, z=z, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, arguments=arguments);
        }
        
        if(plinear==1){
          fits[[i]]   <- cond_comp(datause=datause1, dataout=datause2, y=y, d=d, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, arguments=arguments);
        }
        
        pred1[,1,i] <- fits[[i]][['my_x']]
        pred1[,2,i] <- fits[[i]][['mz_x']]
        
        if(plinear==2){
          pred1[,3,i] <- fits[[i]][['mz2_x']]
        }
      }
      
      for(p in 1:nrow(weight)){
        
        errorM[p,j,1] <- error(pred1[,1,] %*% (weight[p,]), datause2[,y])$err 
        errorM[p,j,2] <- error(pred1[,2,] %*% (weight[p,]), datause2[,d])$err 
        
        if(plinear==2){
          errorM[p,j,3] <- error(pred1[,3,] %*% (weight[p,]), datause2[,z])$err 
        }
        
      }
    }
    
    min1 <- which.min(as.matrix(rowSums(errorM[,,1])))
    min2 <- which.min(as.matrix(rowSums(errorM[,,2])))
    
    if(plinear==2){
      min3 <- which.min(as.matrix(rowSums(errorM[,,3])))
    }
    
    for(i in 1:k){
      
      if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==method[i])){
        x=xL
      } else {
        x=xx
      }
      
      if(plinear==2){
        fits[[i]]   <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, z=z, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, arguments=arguments);
      }
      
      if(plinear==1){
        fits[[i]]   <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, arguments=arguments);
      }
      
      pred2[,1,i] <- fits[[i]][['my_x']]
      pred2[,2,i] <- fits[[i]][['mz_x']]
      
      if(plinear==2){
        pred2[,3,i] <- fits[[i]][['mz2_x']]
      }
      
    }
    
    fit.y  <- pred2[,1,] %*% (weight[min1,])
    fit.z  <- pred2[,2,] %*% (weight[min2,])
    
    ry     <- dataout[,y] - fit.y
    rz     <- dataout[,d] - fit.z
    
    err.y  <- error(fit.y, dataout[,y])$err 
    err.z  <- error(fit.z, dataout[,d])$err 
    
    if(plinear==2){
      fit.z2 <- pred2[,3,] %*% (weight[min3,])
      rz2 <- dataout[,z] - fit.z2
      err.z2 <- error(fit.z2, dataout[,z])$err 
    }
    
    return(list(err.z=err.z, err.y=err.y, err.z2=err.z2, ry=ry, rz=rz, rz2=rz2));
    
  }
  
  
  if(plinear==3){
    
    errorM  <- array(0,dim=c(nrow(weight),5,5))
    pred2   <- array(0,dim=c(nrow(dataout),5,k))  
    
    for(j in 1:K){
      
      ii   <- cvgroup == j
      nii  <- cvgroup != j
      
      datause1 <- as.data.frame(datause[nii,])
      datause2 <- as.data.frame(datause[ii,])  
      pred1    <- array(0,dim=c(nrow(datause2),5,k))  
      
      ind_u1     <- which(datause1[,d]==1)
      ind_u2     <- which(datause2[,d]==1)
      if(plinear==3){
        ind_u1     <- which(datause1[,z]==1)
        ind_u2     <- which(datause2[,z]==1)
      }
      
      for(i in 1:k){
        
        if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==method[i])){
          x=xL
        } else {
          x=xx
        }
        
        fits[[i]]   <- cond_comp(datause=datause1, dataout=datause2, y=y, d=d, z=z, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, flag=flag, arguments=arguments);
        
        pred1[,1,i] <- fits[[i]][['my_z1x']]
        pred1[,2,i] <- fits[[i]][['my_z0x']]
        pred1[,3,i] <- fits[[i]][['md_z1x']]
        if(flag==1){ pred1[,4,i] <- matrix(0,1,length(fits[[i]][['md_z1x']]))}
        else{pred1[,4,i] <- fits[[i]][['md_z0x']]}
        pred1[,5,i] <- fits[[i]][['mz2_x']]
        
      }
      
      for(p in 1:nrow(weight)){
        
        errorM[p,j,1] <- error(pred1[ind_u2,1,] %*% (weight[p,]), datause2[ind_u2,y])$err 
        errorM[p,j,2] <- error(pred1[-ind_u2,2,] %*% (weight[p,]), datause2[-ind_u2,y])$err 
        errorM[p,j,3] <- error(pred1[ind_u2,3,] %*% (weight[p,]), datause2[ind_u2,d])$err 
        errorM[p,j,4] <- error(pred1[-ind_u2,4,] %*% (weight[p,]), datause2[-ind_u2,d])$err 
        errorM[p,j,5] <- error(pred1[,5,] %*% (weight[p,]), datause2[,z])$err 
        
      }
    }
    
    min1 <- which.min(as.matrix(rowSums(errorM[,,1])))
    min2 <- which.min(as.matrix(rowSums(errorM[,,2])))
    min3 <- which.min(as.matrix(rowSums(errorM[,,3])))
    min4 <- which.min(as.matrix(rowSums(errorM[,,4])))
    min5 <- which.min(as.matrix(rowSums(errorM[,,5])))
    
    for(i in 1:k){
      
      if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==method[i])){
        x=xL
      } else {
        x=xx
      }
      
      fits[[i]] <-cond_comp(datause=datause, dataout=dataout, y=y, d=d, z=z, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary,flag=flag, arguments=arguments);
      
      pred2[,1,i] <- fits[[i]][['my_z1x']]
      pred2[,2,i] <- fits[[i]][['my_z0x']]
      pred2[,3,i] <- fits[[i]][['md_z1x']]
      if(flag==1){ pred2[,4,i] <- matrix(0,1,length(fits[[i]][['md_z1x']]))}
      else{pred2[,4,i] <- matrix(0,1,fits[[i]][['md_z0x']])}
      pred2[,5,i] <- fits[[i]][['mz2_x']]
      
    }
    
    fit.yz1x   <- pred2[,1,] %*% (weight[min1,])
    fit.yz0x   <- pred2[,2,] %*% (weight[min2,])
    fit.dz1x   <- pred2[,3,] %*% (weight[min3,])
    fit.dz0x   <- pred2[,4,] %*% (weight[min4,])
    fit.z2x    <- pred2[,5,] %*% (weight[min5,])
    
    err.yz0  <- error(fit.yz1x[-ind_o], dataout[-ind_o,y])$err 
    err.yz1  <- error(fit.yz0x[ind_o], dataout[ind_o,y])$err 
    err.dz0  <- error(fit.dz0x[-ind_o], dataout[-ind_o,d])$err 
    err.dz1  <- error(fit.dz1x[ind_o], dataout[ind_o,d])$err 
    err.z2   <- error(fit.z2x, dataout[,z])$err 
    
    return(list(err.yz0=err.yz0, err.yz1=err.yz1, err.dz0=err.dz0, err.dz1=err.dz1, err.z2=err.z2, my_z0x=fit.yz0x, my_z1x=fit.yz1x, md_z0x=fit.dz0x, md_z1x=fit.dz1x, mz2_x=fit.z2x));
    
  }
  
  
  if(plinear==0){
    
    errorM  <- array(0,dim=c(nrow(weight),5,3))
    pred2   <- array(0,dim=c(nrow(dataout),3,k))  
    pred3   <- matrix(0,nrow(dataout[ind_o,]),k)
    pred4   <- matrix(0,nrow(dataout[-ind_o,]),k)
    
    for(j in 1:K){
      
      ii = cvgroup == j
      nii = cvgroup != j
      
      datause1 = as.data.frame(datause[nii,])
      datause2 = as.data.frame(datause[ii,])  
      
      ind2_u   <- which(datause1[,d]==1)
      ind2_o   <- which(datause2[,d]==1)
      
      pred1  <- array(0,dim=c(nrow(datause2),3,k))  
      
      for(i in 1:k){
        
        if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==method[i])){
          x=xL
        } else {
          x=xx
        }
        
        fits[[i]] <-cond_comp(datause=datause1, dataout=datause2, y=y, d=d, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, arguments=arguments);
        
        pred1[,1,i] <- fits[[i]][['my_z1x']]
        pred1[,2,i] <- fits[[i]][['my_z0x']]
        pred1[,3,i] <- fits[[i]][['mz_x']]
      }
      
      for(p in 1:nrow(weight)){
        
        errorM[p,j,1] <- error(pred1[ind2_o,1,] %*% (weight[p,]), datause2[ind2_o,y])$err 
        errorM[p,j,2] <- error(pred1[-ind2_o,2,] %*% (weight[p,]), datause2[-ind2_o,d])$err 
        errorM[p,j,3] <- error(pred1[,3,] %*% (weight[p,]), datause2[,d])$err 
      }
    }
    
    min1 <- which.min(as.matrix(rowSums(errorM[,,1])))
    min2 <- which.min(as.matrix(rowSums(errorM[,,2])))
    min3 <- which.min(as.matrix(rowSums(errorM[,,3])))
    
    for(i in 1:k){
      
      if (any(c("RLasso", "PostRLasso", "Ridge", "Lasso", "Elnet")==method[i])){
        x=xL
      } else {
        x=xx
      }
      
      fits[[i]]   <- cond_comp(datause=datause, dataout=dataout, y=y, d=d, x=x, method=method[i], plinear=plinear, xL=xL, binary=binary, arguments=arguments);
      
      pred2[,1,i] <- fits[[i]][['my_z1x']]
      pred2[,2,i] <- fits[[i]][['my_z0x']]
      pred2[,3,i] <- fits[[i]][['mz_x']]
      pred3[,i]   <- fits[[i]][['fit.yz1out']]
      pred4[,i]   <- fits[[i]][['fit.yz0out']]
      
    }
    
    fit.y1x    <- pred2[,1,] %*% (weight[min1,])
    fit.y0x    <- pred2[,2,] %*% (weight[min2,])
    fit.zx     <- pred2[,3,] %*% (weight[min3,])
    fit.yz1out <- pred3 %*% (weight[min1,])
    fit.yz0out <- pred4 %*% (weight[min2,])
    
    err.y1x <- error(fit.yz1out, dataout[ind_o,y])$err 
    err.y0x <- error(fit.yz0out, dataout[-ind_o,y])$err 
    err.z   <- error(fit.zx, dataout[,d])$err 
    
    
    return(list(err.z=err.z, err.yz1=err.y1x, err.yz0=err.y0x,  my_z1x=fit.y1x, my_z0x=fit.y0x,mz_x=fit.zx ));
    
  }
}



