library(R2jags)
library(coda)
library(ggplot2)

source("bayesianUtilities.R")

####################################################################################
#### Data without outliers
####################################################################################

##¡¡Data

# Northern rockfish male data (from Quinn and Deriso 1999)

growth <- read.csv("Growithoutoutliers.csv")
ggplot(data = growth, aes(x = Age, y = Length)) + geom_point() + theme_bw()

x <- growth$Age
y <- growth$Length
n <- nrow(growth)

my.data <- list("x","y","n")

###########################################################################################
## Likelihood function is normal.
###########################################################################################

# Model

my.model <- function() {
  # Likelihood
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dnorm(eta[i], tau)
  }
  
  # Prior for the error precision (=1/Sigma^2)
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
}

# Params

my.params<- c("Linf", "k", "t0", "sigma")

# Inits

my.inits <- function() {
  list("Linf" = dunif(1, 0, 10000),
       "k" = dunif(1, 0, 10),
       "t0" = dunif(1, -100, 100),
       "sigma" = dunif(1, 0.01, 100))
}

# Run model

fit.1 <- jags(data=my.data, inits=my.inits, parameters.to.save=my.params, 
              model.file=my.model, n.iter=5000, n.burnin=1000, n.thin=1, n.chains=3)
mcmc.1 <- as.mcmc(fit.1)

traceplot(fit.1)

raftery.diag(mcmc.1[[1]], q=0.025) 
raftery.diag(mcmc.1[[1]], q=0.975) 
# Need a much longer chain. 4000*15+1000=61000

gelman.diag(mcmc.1) # Down convergence

crosscorr(mcmc.1) # Strong correlations between "k" and "Linf", between "k" and "t0", between "Linf" and "t0".

# Longer chain

fit.2 <- jags(data=my.data, inits=my.inits, parameters.to.save=my.params, 
              model.file=my.model, n.iter=61000, n.burnin=1000, n.thin=15, n.chains=3)
mcmc.2 <- as.mcmc(fit.2)

traceplot(fit.2)

raftery.diag(mcmc.2[[1]], q=0.025) 
raftery.diag(mcmc.2[[1]], q=0.975) 
# Still need a longer chain

gelman.diag(mcmc.2) # Good

crosscorr(mcmc.2) # Strong correlations between "k" and "Linf", between "k" and "t0", between "Linf" and "t0".

summary(mcmc.2, quantile=c(0.025, 0.975))

## Glue the three chains
glue.chains <- mcmc.list.to.single.mcmc(mcmc.2)
HPDinterval(glue.chains, prob=0.99) 
summary(mcmc.2, quantile=c(0.005, 0.995))
# The 99% HPD interval is narrower than the central 99% interval.

## Calculate omnibus statistic

my.model.p <- function() {
  #  Likelihood 
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dnorm(eta[i], tau)
    T[i] <- (y[i] - eta[i])^2 * tau
    y.rep[i] ~ dnorm(eta[i], tau) # generate data using the posterior
    T.rep[i] <- (y.rep[i] - eta[i])^2 * tau
  }
  
  # Priors
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
}

# Params

my.params.p <- c("Linf", "k", "t0", "sigma","T", "T.rep", "y.rep")

## Run model

fit.p <- jags(data = my.data, parameters.to.save = my.params.p, 
                      model.file = my.model.p, n.iter = 61000, n.burnin=1000, n.thin=15, n.chains=3)

T.mx <- fit.p$BUGSoutput$sims.list$T 
T.rep.mx <- fit.p$BUGSoutput$sims.list$T.rep
# Sum across rows of T and T.rep to compute T for each iteration
T.all <- apply(T.mx, 1, sum)
T.rep.all <- apply(T.rep.mx, 1, sum)
# Now compute how often T.rep.all > T.all:
(bayes.p.value <- sum(T.rep.all > T.all)/length(T.all))


## Check outliers: Bayesian p-value, point by point

bayes.p.by.point <- numeric(n)
for (i in 1:n) {bayes.p.by.point[i] <- sum(T.rep.mx[,i] > T.mx[,i])/nrow(T.mx)}
df.for.gg.1 <- data.frame(x=growth$Age, y=growth$Length, p=bayes.p.by.point)
ggplot(df.for.gg.1, aes(x=x, y=y, color=p)) + geom_point() + theme_bw() + 
  xlab("Age") + ylab("Length") + scale_colour_gradientn(colours=rainbow(7)) 

## There are some red points fitting badly.

## Prediction intervals

q.025 <- function(x) {quantile(x, 0.025)}
q.975 <- function(x) {quantile(x, 0.975)}
all.y.rep <- fit.p$BUGSoutput$sims.list$y.rep
plot.df <- data.frame(age=jitter(growth$Age), length=growth$Length, 
                      ymin=apply(all.y.rep, 2, q.025),
                      ymid=apply(all.y.rep, 2, median),
                      ymax=apply(all.y.rep, 2, q.975))
ggplot(data=plot.df, aes(x=age, y=ymid, ymin=ymin, ymax=ymax)) + 
  geom_pointrange() + theme_bw() + geom_point(aes(y=length), col="red") + xlab("Age") + ylab("Length")

## Some outliers.

fit.p$BUGSoutput$pD  
fit.p$BUGSoutput$DIC   

###########################################################################################
## Likelihood function is not normal, with thicker tails.
###########################################################################################
# Model

my.model.t<- function() {
  # Likelihood
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dt(eta[i], tau, tdf)
  }
  
  # Prior for the error precision (=1/Sigma^2)
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
  tdf ~ dunif(2,100)
}

# Params

my.params.t<- c("Linf", "k", "t0", "sigma", "tdf")

# Inits

my.inits.t <- function() {
  list("Linf" = dunif(1, 0, 10000),
       "k" = dunif(1, 0, 10),
       "t0" = dunif(1, -100, 100),
       "sigma" = dunif(1, 0.01, 100),
       "tdf" = dunif(1, 2, 100))
}

# Run model

fit.t <- jags(data=my.data, inits=my.inits.t, parameters.to.save=my.params.t, 
              model.file=my.model.t, n.iter=5000, n.burnin=1000, n.thin=1, n.chains=3)
mcmc.t <- as.mcmc(fit.t)

traceplot(fit.t)

raftery.diag(mcmc.t[[1]], q=0.025) 
raftery.diag(mcmc.t[[1]], q=0.975) 
# Need a much longer chain. 4000*21+1000=85000

gelman.diag(mcmc.t) # Down convergence!

crosscorr(mcmc.t) # Strong correlations between "k" and "Linf", between "k" and "t0".

# Longer chain

fit.t.2 <- jags(data=my.data, inits=my.inits.t, parameters.to.save=my.params.t, 
              model.file=my.model.t, n.iter=85000, n.burnin=1000, n.thin=21, n.chains=3)
mcmc.t.2 <- as.mcmc(fit.t.2)

traceplot(fit.t.2)

raftery.diag(mcmc.t.2[[1]], q=0.025) 
raftery.diag(mcmc.t.2[[1]], q=0.975) 
# Still need a longer chain

gelman.diag(mcmc.t.2) # Good

crosscorr(mcmc.t.2) # Strong correlations between "k" and "Linf", between "k" and "t0", between "Linf" and "t0".

summary(mcmc.t.2, quantile=c(0.025, 0.975))

## Glue the three chains
glue.chains.t <- mcmc.list.to.single.mcmc(mcmc.t.2)
HPDinterval(glue.chains.t, prob=0.99) 
summary(mcmc.t.2, quantile=c(0.005, 0.995))
# The 99% HPD interval is narrower than the central 99% interval.

## Calculate omnibus statistic

my.model.t.p <- function() {
  #  Likelihood 
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dt(eta[i], tau, tdf)
    T[i] <- (y[i] - eta[i])^2 * tau
    y.rep[i] ~ dt(eta[i], tau, tdf) # generate data using the posterior
    T.rep[i] <- (y.rep[i] - eta[i])^2 * tau
  }
  
  # Priors
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
  tdf ~ dunif(2,100)
}

# Params

my.params.t.p <- c("Linf", "k", "t0", "sigma","tdf", "T", "T.rep", "y.rep")

## Run model

fit.t.p <- jags(data = my.data, parameters.to.save = my.params.t.p, 
              model.file = my.model.t.p, n.iter = 85000, n.burnin=1000, n.thin=21, n.chains=3)

T.mx <- fit.t.p$BUGSoutput$sims.list$T 
T.rep.mx <- fit.t.p$BUGSoutput$sims.list$T.rep
# Sum across rows of T and T.rep to compute T for each iteration
T.all <- apply(T.mx, 1, sum)
T.rep.all <- apply(T.rep.mx, 1, sum)
# Now compute how often T.rep.all > T.all:
(bayes.p.value <- sum(T.rep.all > T.all)/length(T.all))


## Check outliers: Bayesian p-value, point by point

bayes.p.by.point <- numeric(n)
for (i in 1:n) {bayes.p.by.point[i] <- sum(T.rep.mx[,i] > T.mx[,i])/nrow(T.mx)}
df.for.gg.1 <- data.frame(x=growth$Age, y=growth$Length, p=bayes.p.by.point)
ggplot(df.for.gg.1, aes(x=x, y=y, color=p)) + geom_point() + theme_bw() + 
  xlab("Age") + ylab("Length") + scale_colour_gradientn(colours=rainbow(7)) 

## There are some red points fitting badly 

## Prediction intervals

q.025 <- function(x) {quantile(x, 0.025)}
q.975 <- function(x) {quantile(x, 0.975)}
all.y.rep <- fit.t.p$BUGSoutput$sims.list$y.rep
plot.df <- data.frame(age=jitter(growth$Age), length=growth$Length, 
                      ymin=apply(all.y.rep, 2, q.025),
                      ymid=apply(all.y.rep, 2, median),
                      ymax=apply(all.y.rep, 2, q.975))
ggplot(data=plot.df, aes(x=age, y=ymid, ymin=ymin, ymax=ymax)) + 
  geom_pointrange() + theme_bw() + geom_point(aes(y=length), col="red") + xlab("Age") + ylab("Length")

## Some outliers 

fit.t.p$BUGSoutput$pD   
fit.t.p$BUGSoutput$DIC  

####################################################################################
#### Data with outliers
####################################################################################

##¡¡Data

# Northern rockfish male data (from Quinn and Deriso 1999)

dataout <- read.csv("Growthwithoutliers.csv")
ggplot(data = dataout, aes(x = Age, y = Length)) + geom_point() + theme_bw()

x <- dataout$Age
y <- dataout$Length
n <- nrow(dataout)

my.data.out <- list("x","y","n")

###########################################################################################
## Likelihood function is normal.
###########################################################################################

# Model

my.model.out <- function() {
  # Likelihood
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dnorm(eta[i], tau)
  }
  
  # Prior for the error precision (=1/Sigma^2)
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
}

# Params

my.params.out<- c("Linf", "k", "t0", "sigma")

# Inits

my.inits.out <- function() {
  list("Linf" = dunif(1, 0, 10000),
       "k" = dunif(1, 0, 10),
       "t0" = dunif(1, -100, 100),
       "sigma" = dunif(1, 0.01, 100))
}

# Run model

fit.1.out <- jags(data=my.data.out, inits=my.inits.out, parameters.to.save=my.params.out, 
                  model.file=my.model.out, n.iter=5000, n.burnin=1000, n.thin=1, n.chains=3)
mcmc.1.out <- as.mcmc(fit.1.out)

traceplot(fit.1.out)

raftery.diag(mcmc.1.out[[1]], q=0.025) 
raftery.diag(mcmc.1.out[[1]], q=0.975) 
# Need a much longer chain. 4000*18+1000=73000

gelman.diag(mcmc.1.out) # Down convergence

crosscorr(mcmc.1.out) # Strong correlations between "k" and "Linf", between "k" and "t0", between "Linf" and "t0".

# Longer chain

fit.2.out <- jags(data=my.data.out, inits=my.inits.out, parameters.to.save=my.params.out, 
                  model.file=my.model.out, n.iter=73000, n.burnin=1000, n.thin=18, n.chains=3)
mcmc.2.out <- as.mcmc(fit.2.out)

traceplot(fit.2.out)

raftery.diag(mcmc.2.out[[1]], q=0.025) 
raftery.diag(mcmc.2.out[[1]], q=0.975) 
# Still need a longer chain

gelman.diag(mcmc.2.out) # Good

crosscorr(mcmc.2.out) # Strong correlations between "k" and "Linf", between "k" and "t0", between "Linf" and "t0".

summary(mcmc.2.out, quantile=c(0.025, 0.975))


## Glue the three chains
glue.chains.out <- mcmc.list.to.single.mcmc(mcmc.2.out)
HPDinterval(glue.chains.out, prob=0.99) 
summary(mcmc.2.out, quantile=c(0.005, 0.995))
# The 99% HPD interval is narrower than the central 99% interval.

## Calculate omnibus statistic

my.model.out.p <- function() {
  #  Likelihood 
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dnorm(eta[i], tau)
    T[i] <- (y[i] - eta[i])^2 * tau
    y.rep[i] ~ dnorm(eta[i], tau) # generate data using the posterior
    T.rep[i] <- (y.rep[i] - eta[i])^2 * tau
  }
  
  # Priors
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
}

# Params

my.params.out.p <- c("Linf", "k", "t0", "sigma","T", "T.rep", "y.rep")

## Run model

fit.out.p <- jags(data = my.data.out, parameters.to.save = my.params.out.p, 
                  model.file = my.model.out.p, n.iter = 73000, n.burnin=1000, n.thin=18, n.chains=3)

T.mx <- fit.out.p$BUGSoutput$sims.list$T 
T.rep.mx <- fit.out.p$BUGSoutput$sims.list$T.rep
# Sum across rows of T and T.rep to compute T for each iteration
T.all <- apply(T.mx, 1, sum)
T.rep.all <- apply(T.rep.mx, 1, sum)
# Now compute how often T.rep.all > T.all:
(bayes.p.value <- sum(T.rep.all > T.all)/length(T.all))
# 0.5200833, good

## Check outliers: Bayesian p-value, point by point

bayes.p.by.point <- numeric(n)
for (i in 1:n) {bayes.p.by.point[i] <- sum(T.rep.mx[,i] > T.mx[,i])/nrow(T.mx)}
df.for.gg.1 <- data.frame(x=dataout$Age, y=dataout$Length, p=bayes.p.by.point)
ggplot(df.for.gg.1, aes(x=x, y=y, color=p)) + geom_point() + theme_bw() + 
  xlab("Age") + ylab("Length") + scale_colour_gradientn(colours=rainbow(7)) 


## Prediction intervals

q.025 <- function(x) {quantile(x, 0.025)}
q.975 <- function(x) {quantile(x, 0.975)}
all.y.rep <- fit.out.p$BUGSoutput$sims.list$y.rep
plot.df <- data.frame(age=jitter(dataout$Age), length=dataout$Length, 
                      ymin=apply(all.y.rep, 2, q.025),
                      ymid=apply(all.y.rep, 2, median),
                      ymax=apply(all.y.rep, 2, q.975))
ggplot(data=plot.df, aes(x=age, y=ymid, ymin=ymin, ymax=ymax)) + 
  geom_pointrange() + theme_bw() + geom_point(aes(y=length), col="red") + xlab("Age") + ylab("Length")

## Some outliers.

fit.out.p$BUGSoutput$pD   # 467.9954
fit.out.p$BUGSoutput$DIC   # 2233.907

###########################################################################################
## Likelihood function is not normal, with thicker tails.
###########################################################################################
# Model

my.model.t.out <- function() {
  # Likelihood
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dt(eta[i], tau, tdf)
  }
  
  # Prior for the error precision (=1/Sigma^2)
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
  tdf ~ dunif(2,100)
}

# Params

my.params.t.out <- c("Linf", "k", "t0", "sigma", "tdf")

# Inits

my.inits.t.out <- function() {
  list("Linf" = dunif(1, 0, 10000),
       "k" = dunif(1, 0, 10),
       "t0" = dunif(1, -100, 100),
       "sigma" = dunif(1, 0.01, 100),
       "tdf" = dunif(1, 2, 100))
}

# Run model

fit.t.out <- jags(data=my.data.out, inits=my.inits.t.out, parameters.to.save=my.params.t.out, 
                  model.file=my.model.t.out, n.iter=5000, n.burnin=1000, n.thin=1, n.chains=3)
mcmc.t.out <- as.mcmc(fit.t.out)

traceplot(fit.t.out)

raftery.diag(mcmc.t.out[[1]], q=0.025) 
raftery.diag(mcmc.t.out[[1]], q=0.975) 
# Need a much longer chain. 4000*26+1000=105000

gelman.diag(mcmc.t.out) # Down convergence!

crosscorr(mcmc.t.out) # Strong correlations between "k" and "Linf", between "k" and "t0".

# Longer chain

fit.t.2.out <- jags(data=my.data.out, inits=my.inits.t.out, parameters.to.save=my.params.t.out, 
                    model.file=my.model.t.out, n.iter=105000, n.burnin=1000, n.thin=26, n.chains=3)
mcmc.t.2.out <- as.mcmc(fit.t.2.out)

traceplot(fit.t.2.out)

raftery.diag(mcmc.t.2.out[[1]], q=0.025) 
raftery.diag(mcmc.t.2.out[[1]], q=0.975) 
# Still need a longer chain

gelman.diag(mcmc.t.2.out) # Good

crosscorr(mcmc.t.2.out) # Strong correlations between "k" and "Linf", between "k" and "t0", between "Linf" and "t0".

summary(mcmc.t.2.out, quantile=c(0.025, 0.975))

## Glue the three chains
glue.chains.t.out <- mcmc.list.to.single.mcmc(mcmc.t.2.out)
HPDinterval(glue.chains.t.out, prob=0.99) 
summary(mcmc.t.2.out, quantile=c(0.005, 0.995))
# The 99% HPD interval is narrower than the central 99% interval.

## Calculate omnibus statistic

my.model.t.out.p <- function() {
  #  Likelihood 
  for (i in 1:n) {
    eta[i] <- Linf * (1-exp(-k * (x[i] - t0)))
    y[i] ~ dt(eta[i], tau, tdf)
    T[i] <- (y[i] - eta[i])^2 * tau
    y.rep[i] ~ dt(eta[i], tau, tdf) # generate data using the posterior
    T.rep[i] <- (y.rep[i] - eta[i])^2 * tau
  }
  
  # Priors
  Linf ~ dunif(0, 10000)
  k ~ dunif(0, 10)     
  t0 ~ dunif(-100, 100)
  tau <- 1/sigma/sigma
  sigma ~ dunif(0.01, 100)
  tdf ~ dunif(2,100)
}

# Params

my.params.t.out.p <- c("Linf", "k", "t0", "sigma","tdf", "T", "T.rep", "y.rep")

## Run model

fit.t.out.p <- jags(data = my.data.out, parameters.to.save = my.params.t.out.p, 
              model.file = my.model.t.out.p, n.iter = 105000, n.burnin=1000, n.thin=26, n.chains=3)

T.mx <- fit.t.out.p$BUGSoutput$sims.list$T 
T.rep.mx <- fit.t.out.p$BUGSoutput$sims.list$T.rep
# Sum across rows of T and T.rep to compute T for each iteration
T.all <- apply(T.mx, 1, sum)
T.rep.all <- apply(T.rep.mx, 1, sum)
# Now compute how often T.rep.all > T.all:
(bayes.p.value <- sum(T.rep.all > T.all)/length(T.all))
## 0.2449167, which means the realizations from the model don¡¯t have enough emphasis on the tails 
## of the distribution. It makes sense, since the model is robust to outliers.

## Check outliers: Bayesian p-value, point by point

bayes.p.by.point <- numeric(n)
for (i in 1:n) {bayes.p.by.point[i] <- sum(T.rep.mx[,i] > T.mx[,i])/nrow(T.mx)}
df.for.gg.1 <- data.frame(x=dataout$Age, y=dataout$Length, p=bayes.p.by.point)
ggplot(df.for.gg.1, aes(x=x, y=y, color=p)) + geom_point() + theme_bw() + 
  xlab("Age") + ylab("Length") + scale_colour_gradientn(colours=rainbow(7)) 

## There are some red points fitting badly.

## Prediction intervals

q.025 <- function(x) {quantile(x, 0.025)}
q.975 <- function(x) {quantile(x, 0.975)}
all.y.rep <- fit.t.out.p$BUGSoutput$sims.list$y.rep
plot.df <- data.frame(age=jitter(dataout$Age), length=dataout$Length, 
                      ymin=apply(all.y.rep, 2, q.025),
                      ymid=apply(all.y.rep, 2, median),
                      ymax=apply(all.y.rep, 2, q.975))
ggplot(data=plot.df, aes(x=age, y=ymid, ymin=ymin, ymax=ymax)) + 
  geom_pointrange() + theme_bw() + geom_point(aes(y=length), col="red") + xlab("Age") + ylab("Length")

## Some outliers

fit.t.out.p$BUGSoutput$pD   
fit.t.out.p$BUGSoutput$DIC 

######################################################################################

## Plot

par(mfrow =c(2,2))
plot(density(glue.chains[,"Linf"]),xlim=c(350,410), ylim=c(0,0.1),lwd=3)
lines(density(glue.chains.t[,"Linf"]), lty=2, lwd=3, col="blue")
lines(density(glue.chains.out[,"Linf"]), lty=3, lwd=3, col="red")
lines(density(glue.chains.t.out[,"Linf"]), lty=3, lwd=3, col="green")
legend(380,0.1,c("Normal + no outliers","Robust + no outliers","Normal + outliers", "Robust + outliers"),
lty=c(1,2,3,3),lwd=c(3,3,3,3),cex=1.05,bty="n",col=c("black", "blue", "red", "green"))

plot(density(glue.chains[,"k"]), xlim=c(0.05,0.41),lwd=3)
lines(density(glue.chains.t[,"k"]), lty=2, lwd=3, col="blue")
lines(density(glue.chains.out[,"k"]), lty=3, lwd=3, col="red")
lines(density(glue.chains.t.out[,"k"]), lty=3, lwd=3, col="green")

plot(density(glue.chains[,"t0"]), xlim=c(-10,5), ylim=c(0,0.5),lwd=3)
lines(density(glue.chains.t[,"t0"]), lty=2, lwd=3, col="blue")
lines(density(glue.chains.out[,"t0"]), lty=3, lwd=3, col="red")
lines(density(glue.chains.t.out[,"t0"]), lty=3, lwd=3, col="green")

plot(density(glue.chains[,"sigma"]), xlim=c(12,30),lwd=3)
lines(density(glue.chains.t[,"sigma"]), lty=2, lwd=3, col="blue")
lines(density(glue.chains.out[,"sigma"]), lty=3, lwd=3, col="red")
lines(density(glue.chains.t.out[,"sigma"]), lty=3, lwd=3, col="green")


