uniHawkesCompensator_old<-function(lambda0,alpha,beta,t){
  # [Incorrect] Due to Typo...
  
  #t shouldn't include the 1st event time
  n<-length(t)
  delta.t<-t-c(0,t[-n])#t[-n]: drop the last element in t
  #delta.t: inter-event time
  Lambda<-rep(0,n)
  A<-0
  Lambda[1]<-lambda0*(t[1])*2
  for(i in 2:n){
    A<-1+exp(-beta*(delta.t[i-1]))*A
    Lambda[i]<-lambda0*(delta.t[i])*2+alpha/beta*(1-exp(-beta*delta.t[i]))*A
  }
  return(Lambda)
}

uniHawkesCompensator.Owen <- function(lambda0,alpha,beta, events){
  #[Owen]
  # input parameters for Hawkes process, include lambda0, alpha, beta
  #       events: vector of event happening time
  # output Lambda: vector of intensity compensator
  
  
  N<-length(events)
  Lambda<-rep(0,N)
  r<-0
  Lambda[1]<-lambda0*(events[1])
  for(i in 2:N){
    delta.t <- events[i]-events[i-1]
    temp.r <-exp(-beta*delta.t)*(r+1)
    Lambda[i]<-lambda0*delta.t-alpha/beta*(temp.r-r-1)
    r <- temp.r
  }
  return(Lambda)
}

uniHawkesCompensator<-function(lambda0,alpha,beta,t){
  #[Diane]
  #[Ref]Analysis of Order Clustering Using High Frequency Data: A Point Process Approach (Eq 2.29)
  #t should include the 1st event time
  #outputs Lambda(t_i,t_{i+1})
  
  n<-length(t)#number of event
  Lambda<-rep(0,n-1)
  
  for(i in 1:(n-1)){
    
    delta.t.next<-t[i+1]-t[1:i]#t[i+1]-t[k] where k=1~i
    delta.t.current <- t[i] - t[1:i] ##t[i]-t[k] where k=1~i
    
    Lambda[i]<-lambda0*(t[i+1]-t[i])-alpha/beta*sum(exp(-beta*delta.t.next)-exp(-beta*delta.t.current))
  }
  return(Lambda)
}

uniHawkesCompensatorPoisson<-function(lambda0,alpha,beta,t){
  #[Diane]
  #[Ref]Analysis of Order Clustering Using High Frequency Data: A Point Process Approach (Eq 2.29)
  #t should include the 1st event time
  #outputs Lambda(t_i) for i=1,...,(n)
  
  n<-length(t)#number of event
  Lambda<-rep(0,n)
  
  dur_res_proc <- uniHawkesCompensator(lambda0,alpha,beta,t)
  Lambda[2:n] <- cumsum(dur_res_proc)
  #Lambda(t_1) =0 because t_1 = 0
  return(Lambda)
}

uniHawkesNegLogLik.constraint <- function(params=list(lambda0,alpha,beta), t) {
  #https://radhakrishna.typepad.com/mle-of-hawkes-self-exciting-process.pdf
  #https://stats.stackexchange.com/questions/24685/finding-the-mle-for-a-univariate-exponential-hawkes-process
  #t should include the first event time
  #constraint: alpha < beta 
  #ref: A Tutorial on Hawkes Processes for Events in Social Media (2017) Eq(1.12)
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  n <- length(t)
  r <- rep(0,n)
  if(alpha < beta){#[Diane]constraint on parameter space
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
  }else{
    loglik<--1e+10 
  }
  
  return(-loglik)
}

#can be sensitive to initial value(change lambda0 initial value from 0.01 to 0.1, folders 1,2,19 results in very differently)
uniHawkesNegLogLik <- function(params=list(lambda0,alpha,beta), t) {
  #https://radhakrishna.typepad.com/mle-of-hawkes-self-exciting-process.pdf
  #https://stats.stackexchange.com/questions/24685/finding-the-mle-for-a-univariate-exponential-hawkes-process
  #t should include the first event time
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
  #calculate conditional intensity
  intens <- 0
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  keep_times <- times[which(times < t0)]
  intens <- intens + lambda0 + alpha * sum(exp(-beta*(t0-keep_times)))
  return(intens)
}



Res_Hawkes_old <- function(params = list(lambda0,alpha,beta),times, t0){
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  tmp <- uniHawkesCompensator(lambda0,alpha,beta,times)
  n <- length(which(times < t0))
  return(n-sum(tmp))
}

Raw_Res_Hawkes <- function(compen,N_t){
  #[Diane]
  #Residual analysis for spatial point processes (2005)
  #R(t) = N_t - integrate_{0}^{t} lambda_(u) du
  #N_t: number of arrivals in time [0,t] and lambda_(u) is the conditional intensity given the history up to time u
  
  #when t>t1 and assuming t1=0:the latter term can be considered as the sum of {compensator of unit rate} up to time t 
  #else: the latter term = 0
  res <- N_t - c(0,cumsum(compen))
  return(res)
}

sqrt_cond_inten <- function(x, lambda0,alpha,beta, times){
  #x might be a list
  #sqrt of conditional intensity as integrand
  
  #for debugging
  #print("x:")
  #print(x)
  
  cond_int <- rep(0,length(x))
  for(i in 1:length(x)){
    cond_int[i] <- uniHawkesntensity(params=list(lambda0,alpha,beta), times, t0=x[i])
  }
  
  #print(cond_int)
  
  return(sqrt(cond_int))
}
  
Pearson_Res_Hawkes <- function(lambda0,alpha,beta,times){
  #times include 1st observed event
  n <- length(times)
  pearson_res <- rep(0,n-1)
  for(i in 1:(n-1)){
    first_term_denom<- sqrt_cond_inten(times[i+1], lambda0,alpha,beta, times) #sqrt(lambda(t_{i+1}))
    second_term <- integrate(sqrt_cond_inten, lower = times[i], upper=times[i+1], lambda0=lambda0, alpha=alpha, beta=beta, times=times)$value
    pearson_res[i] <- 1/(times[i+1]-times[i])*(1/first_term_denom - second_term)
  }
  return(pearson_res)
  
}

uniHawkes_max_intensity <- function(param_object,times,start,end){
  max_int <- 0
  for(t in start:end){
    int_t <- uniHawkesntensity(param_object, times, t)#intensity at t
    if(int_t > max_int){
      max_int <- int_t
    }
  }
  return(max_int)
}
