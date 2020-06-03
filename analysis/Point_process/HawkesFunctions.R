uniHawkesCompensator<-function(lambda0,alpha,beta,t){
  n<-length(t)
  delta.t<-t-c(0,t[-n])
  Lambda<-rep(0,n)
  A<-0
  Lambda[1]<-lambda0*(t[1])*2
  for(i in 2:n){
    A<-1+exp(-beta*(delta.t[i-1]))*A
    Lambda[i]<-lambda0*(delta.t[i])*2+alpha/beta*(1-exp(-beta*delta.t[i]))*A
  }
  return(Lambda)
}

uniHawkesNegLogLik <- function(params=list(lambda0,alpha,beta), t) {
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  n <- length(t)
  r <- rep(0,n)
  for(i in 2:n) {
    r[i] <- exp(-beta*(t[i]-t[i-1]))*(1+r[i-1])
  }
  loglik <- -t[n]*lambda0
  loglik <- loglik+alpha/beta*sum(exp(-beta*(t[n]-t))-1)
  if(any(lambda0+alpha*r<=0)){
    loglik<--1e+10 
  }else{
    loglik <- loglik+sum(log(lambda0+alpha*r))
  }
  return(-loglik)
}

uniHawkesntensity <- function(params=list(lambda0,alpha,beta), times, t0) {
  intens <- 0
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  keep_times <- times[which(times < t0)]
  intens <- intens + lambda0 + alpha * sum(exp(-beta*(t0-keep_times)))
  return(intens)
}



Res_Hawkes <- function(params = list(lambda0,alpha,beta),times, t0){
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  tmp <- uniHawkesCompensator(lambda0,alpha,beta,times)
  n <- length(which(times < t0))
  return(n-sum(tmp))
}
