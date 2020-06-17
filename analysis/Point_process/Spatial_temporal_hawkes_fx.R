#spatial-temporal
STHawkesIntensity <- function(params=list(lambda0,alpha,beta,gamma), times,locations, t,s) {
  #calculate conditional intensity lambda(s,t)
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  gamma <- params[[4]]
  keep_times <- times[which(times < t)]
  keep_locations <- locations[which(times < t)]
  intens <- lambda0 + alpha * sum(exp(-beta*(t-keep_times))*exp(-gamma*(abs(s-keep_locations))))
  return(intens)
}


STHawkesNegLogLik <- function(params=list(lambda0,alpha,beta,gamma), t, s) {
  #A Review of Self-Exciting Spatio-Temporal Point Processes and Their Applications Eq(8)
  #t should include the first event time
  #t : time, s: location [0,1]
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  gamma <- params[[4]]
  n <- length(t)
  
  
  
  #1st term: -integrate_{0}^{t_n} lambda(t) dt
  time_exp_term <- exp(-beta*(t[n]-t))-1
  location_exp_term <- -2 + exp(-gamma*s) + exp(-gamma*(1-s))
  loglik <- -t[n]*lambda0 - alpha/beta/gamma*sum(time_exp_term*location_exp_term)
  
  #2nd term: integrate_{0}^{t_n} log(lambda(t)) dN(t)
  second_term <- 0
  for(i in 1:n){
    inside_log_i <- lambda0
    
    if (i>=2){
      for(k in 1:(i-1)){
        inside_log_i <- inside_log_i + alpha*exp(-beta*(t[i]-t[k]))*exp(-gamma*(abs(s[i]-s[k])))
      }
    }
    
    if( inside_log_i <= 0 ){#to avoid taking log of non-positive values
      second_to_add_i <- -1e+10 
    }else{
      second_to_add_i <- log(inside_log_i)
    }
    second_term <- second_term + second_to_add_i
    
  }
  loglik <- loglik + second_term
  
  #return "negative" log likelihood
  return(-loglik)
}


#just to double-check that it works the same as uniHawkesNegLogLik in HawkesFunctions.R
TemporalHawkesNegLogLik <- function(params=list(lambda0,alpha,beta), t) {
  #https://radhakrishna.typepad.com/mle-of-hawkes-self-exciting-process.pdf
  #t should include the first event time
  #t : time, s: location [0,1]
  lambda0 <- params[[1]]
  alpha <- params[[2]]
  beta <- params[[3]]
  n <- length(t)
  
  #1st term: -integrate_{0}^{t_n} lambda(t) dt
  time_exp_term <- exp(-beta*(t[n]-t))-1
  loglik <- -t[n]*lambda0 + alpha/beta*sum(time_exp_term)
  
  #2nd term: integrate_{0}^{t_n} log(lambda(t)) dN(t)
  second_term <- 0
  for(i in 1:n){
    inside_log_i <- lambda0
    if (i>=2){
      for(k in 1:(i-1)){
        inside_log_i <- inside_log_i + alpha*exp(-beta*(t[i]-t[k]))
      }
    }
    
    if(inside_log_i<=0){#to avoid taking log of non-positive values
      second_to_add_i <- -1e+10
    }else{
      second_to_add_i <- log(inside_log_i)
    }
    second_term <- second_term + second_to_add_i
    
  }
  loglik <- loglik + second_term
  
  #return "negative" log likelihood
  return(-loglik)
}


STHawkesCompensator<-function(lambda0,alpha,beta,gamma,t,s){
  #t should include the 1st event time
  #outputs Lambda(t_i,t_{i+1}) for i=1,...,(n-1)
  
  n<-length(t)#number of event
  Lambda<-rep(0,n-1)
  
  for(i in 1:(n-1)){
    
    delta.t.next<-t[i+1]-t[1:i]#t[i+1]-t[k] where k=1~i
    delta.t.current <- t[i] - t[1:i] #t[i]-t[k] where k=1~i
    
    delta.s.1k <- 1 - s[1:i]#1-s[k] where k=1~i
    delta.s.k0 <- s[1:i] #s[k] where k=1~i
    
    time_diff_term <- exp(-beta*delta.t.next)-exp(-beta*delta.t.current)
    location_diff_term <- -2 + exp(-gamma*delta.s.k0) + exp(-gamma*delta.s.1k)
    
    Lambda[i]<-lambda0*(t[i+1]-t[i])+alpha/beta/gamma*sum(time_diff_term*location_diff_term)
  }
  return(Lambda)
}

STHawkesCompensatorPoisson<-function(lambda0,alpha,beta,gamma,t,s){
  #http://bemlar.ism.ac.jp/zhuang/pubs/zhuang2018jrssa-manuscript.pdf Eq(49)
  #t should include the 1st event time
  #outputs Lambda(t_i) for i=1,...,(n)
  
  n<-length(t)#number of event
  Lambda<-rep(0,n)
  
  dur_res_proc <- STHawkesCompensator(lambda0,alpha,beta,gamma,t,s)
  Lambda[2:n] <- cumsum(dur_res_proc)
  #Lambda(t_1) =0 because t_1 = 0
  return(Lambda)
}

sqrt_cond_inten_ST <- function(t,s,lambda0,alpha,beta,gamma,times,locations){
  
  #sqrt of conditional intensity as integrand
  cond_int <- STHawkesIntensity(params=list(lambda0,alpha,beta,gamma), times, locations,t,s)
  
  return(sqrt(cond_int))
}

Pearson_Res_STHawkes <- function(lambda0,alpha,beta,gamma,times,locations){
  #Reference:Residual analysis methods for space-time point processes with applications to earthquake forecast models in California (2011) [the eq after Eq(1)]
  #times include 1st observed event
  n <- length(times)
  pearson_res <- rep(0,n-1)
  for(i in 1:(n-1)){
    first_term_denom<- sqrt_cond_inten_ST(times[i+1],locations[i+1], lambda0,alpha,beta,gamma,times,locations) #sqrt(lambda(s_{i+1},t_{i+1}))
    second_term <- integral2(sqrt_cond_inten_ST, xmin = times[i], xmax=times[i+1], ymin=0, ymax=1, vectorized = FALSE,lambda0=lambda0, alpha=alpha, beta=beta, gamma=gamma, times=times, locations=locations)$Q
    #vectorized=FALsE: sqrt_cond_inten_ST can't feed in array/list for input t,s
    
    #for debugging
    # print("first")
    # print(first_term_denom)
    # print("second")
    # print(second_term)
    
    pearson_res[i] <- 1/(times[i+1]-times[i])*(1/first_term_denom - second_term)
  }
  return(pearson_res)
  
}


# #[Incorrect]
# STHawkesNegLogLik.noAbsDiff <- function(params=list(lambda0,alpha,beta,gamma), t, s) {
#   #A Review of Self-Exciting Spatio-Temporal Point Processes and Their Applications Eq(8)
#   #t should include the first event time
#   #t : time, s: location [0,1]
#   lambda0 <- params[[1]]
#   alpha <- params[[2]]
#   beta <- params[[3]]
#   gamma <- params[[4]]
#   n <- length(t)
#   
#   
#   
#   #1st term: -integrate_{0}^{t_n} lambda(t) dt
#   time_exp_term <- exp(-beta*(t[n]-t))-1
#   location_exp_term <- exp(-gamma*(1-s)) -  exp(gamma*s)
#   loglik <- -t[n]*lambda0 - alpha/beta/gamma*sum(time_exp_term*location_exp_term)
#   
#   #2nd term: integrate_{0}^{t_n} log(lambda(t)) dN(t)
#   second_term <- 0
#   for(i in 1:n){
#     inside_log_i <- lambda0
#     
#     if (i>=2){
#       for(k in 1:(i-1)){
#         inside_log_i <- inside_log_i + alpha*exp(-beta*(t[i]-t[k]))*exp(-gamma*(s[i]-s[k]))
#       }
#     }
#     
#     if( inside_log_i <= 0 ){#to avoid taking log of non-positive values
#       second_to_add_i <- -1e+10 
#     }else{
#       second_to_add_i <- log(inside_log_i)
#     }
#     second_term <- second_term + second_to_add_i
#     
#   }
#   loglik <- loglik + second_term
#   
#   #return "negative" log likelihood
#   return(-loglik)
# }
# 
# STHawkesCompensator.noAbsDiff<-function(lambda0,alpha,beta,gamma,t,s){
#   #http://bemlar.ism.ac.jp/zhuang/pubs/zhuang2018jrssa-manuscript.pdf Eq(49)
#   #t should include the 1st event time
#   
#   n<-length(t)#number of event
#   Lambda<-rep(0,n-1)
#   
#   for(i in 1:(n-1)){
#     
#     delta.t.next<-t[i+1]-t[1:i]#t[i+1]-t[k] where k=1~i
#     delta.t.current <- t[i] - t[1:i] #t[i]-t[k] where k=1~i
#     
#     # delta.s.next<-s[i+1]-s[1:i]#s[i+1]-s[k] where k=1~i
#     # delta.s.current <- s[i] - s[1:i] #s[i]-s[k] where k=1~i
#     
#     delta.s.next <- 1 - s[1:i]#1-s[k] where k=1~i
#     delta.s.current <- 0 - s[1:i] #0-s[k] where k=1~i
#     
#     time_diff_term <- exp(-beta*delta.t.next)-exp(-beta*delta.t.current)
#     location_diff_term <- exp(-gamma*delta.s.next)-exp(-gamma*delta.s.current)
#     
#     Lambda[i]<-lambda0*(t[i+1]-t[i])+alpha/beta/gamma*sum(time_diff_term*location_diff_term)
#   }
#   return(Lambda)
# }