#spatial-temporal
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
  location_exp_term <- exp(-gamma*(1-s)) -  exp(gamma*s)
  loglik <- -t[n]*lambda0 - alpha/beta/gamma*sum(time_exp_term*location_exp_term)
  
  #2nd term: integrate_{0}^{t_n} log(lambda(t)) dN(t)
  second_term <- 0
  for(i in 1:n){
    inside_log_i <- lambda0
    
    if (i>=2){
      for(k in 1:(i-1)){
        inside_log_i <- inside_log_i + alpha*exp(-beta*(t[i]-t[k]))*exp(-gamma*(s[i]-s[k]))
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
  #http://bemlar.ism.ac.jp/zhuang/pubs/zhuang2018jrssa-manuscript.pdf Eq(49)
  #t should include the 1st event time
  
  n<-length(t)#number of event
  Lambda<-rep(0,n-1)
  
  for(i in 1:(n-1)){
    
    delta.t.next<-t[i+1]-t[1:i]#t[i+1]-t[k] where k=1~i
    delta.t.current <- t[i] - t[1:i] #t[i]-t[k] where k=1~i
    
    # delta.s.next<-s[i+1]-s[1:i]#s[i+1]-s[k] where k=1~i
    # delta.s.current <- s[i] - s[1:i] #s[i]-s[k] where k=1~i
    
    delta.s.next <- 1 - s[1:i]#s[i+1]-s[k] where k=1~i
    delta.s.current <- 0 - s[1:i] #s[i]-s[k] where k=1~i
    
    time_diff_term <- exp(-beta*delta.t.next)-exp(-beta*delta.t.current)
    location_diff_term <- exp(-gamma*delta.s.next)-exp(-gamma*delta.s.current)
    
    Lambda[i]<-lambda0*(t[i+1]-t[i])+alpha/beta/gamma*sum(time_diff_term*location_diff_term)
  }
  return(Lambda)
}