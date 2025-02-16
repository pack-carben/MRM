# x应该是要留出第一列作为常数项1
##------------------------------------------------------##
##----- mixture of normal linear regression model ------##
##------------------------------------------------------##
mix.Reg.norm.CR.EM <- function(data_y, data_X, data_c, delta, betas = NULL, sigma2 = NULL, 
                            g = NULL, pI = NULL, Class = NULL,
                            error = 0.00001, iter.max = 100, 
                            Stp.rule = c("Log.like", "Atiken"), 
                            per = 1, print = T, fix.sigma = F)
{
  if(isTRUE(print)){
    cat(paste(rep("-", 70), sep = "", collapse = ""), "\n")
    cat('Finite mixture of normally-based linear regression model','\n')
  }
  
  
  # 提取非删失数据
  y <- data_y[delta == 0]  # 非删失的 y
  x <- data_X[delta == 0, ]  # 非删失的 x
  p <- ncol(x); n = length(y) # 计算非删失数据的样本数
  # 提取删失数据
  c <- data_c[delta == 1, ]  # 删失的下界和上界
  cx <- data_X[delta == 1, ]  # 删失对应的自变量
  cn <- nrow(c)  # 不能用length，得用nrow
  
  if(is.null(sigma2) || is.null(betas) || is.null(pI) ){
    
    class = Class
    if(is.null(g) && is.null(class) ) 
      stop("The model is not specified correctly.\n")
    if(!is.null(class)){
      tt = table(Class)
      if(g == 0 || max(as.numeric(labels(tt)$Class)) != g) 
        g = max(as.numeric(labels(tt)$Class))
    }
    
    if(is.null(class)) class = ClusterR::Cluster_Medoids(as.matrix(y), g)$clusters 
    #kmeans(y, g)$cluster
    
    betas = matrix(0, p, g)
    sigma2 = numeric(g)
    for(j in 1:g){
      LM = lm(y[class == j] ~ 1 + x[class == j, 2:p] )
      betas[,j] = LM$coefficients
      sigma2[j] = mean(LM$residuals^2)
    }
    pI = table(class)/n
  }
  
  g = length(pI)
  
  
  ##------------------------------------------------------##
  ##-------- PDF of mix-normal distribution --------------##
  ##------------------------------------------------------##
  d.Norm.mix.reg <- function(y, mu, sigma2, pI, log = F)
  {
    ## y: a vetor of respond observations
    ## mu[, ] a matrix = X^top * beta
    ## sigma2: a vector of sigma2
    g = length(sigma2)
    sigma = sqrt(sigma2)
    PDF <- 0
    for(j in 1:g) PDF <- PDF + pI[j] * dnorm(y, mu[, j], sigma[j], log = F)
    Out = PDF
    Out[which(Out == 0)] <- .Machine$double.xmin
    ifelse(log == T, return(log(Out)), return(Out))
  }
  
  ##------------------------------------------------------##
  ##-------- CDF of mix-normal distribution --------------##
  ##------------------------------------------------------##
  p.Norm.mix.reg <- function(c, cmu, sigma2, pI, log = F)
  {
    ## y: a vetor of respond observations
    ## mu[, ] a matrix = X^top * beta
    ## sigma2: a vector of sigma2
    g = length(sigma2)
    sigma_values = sqrt(sigma2)
    CDF <- 0
    for(j in 1:g){
      CDF <- CDF + pI[j] * (pnorm(c[,2], cmu[, j], sigma_values[j])-
                              pnorm(c[,1], cmu[, j], sigma_values[j]))
    }
    Out = CDF
    Out[which(Out == 0)] <- .Machine$double.xmin #表示可以表示的最小正数的双精度浮点数
    ifelse(log == T, return(log(Out)), return(Out))
  }
  
  ##------------------------------------------------------##
  ##-------- CR y_expected --------------##
  ##------------------------------------------------------##
  e.Norm.mix.cr.y <- function(c,cx,cn,g, betas, sigma2)
  {
    sigma_values = sqrt(sigma2)
    expected_y = matrix(0,cn,g)
    mu = cx %*% betas
    for(j in 1:g){
      # 期望y的分子和分母
      differ_n <- sigma2[j]*(dnorm(c[,1],mu[,j],sigma_values[j]) - 
                              dnorm(c[,2],mu[,j],sigma_values[j]))
      differ_d <- pnorm(c[,2],mu[,j],sigma_values[j]) - dnorm(c[,1],mu[,j],sigma_values[j])
      expected_y[,j] <- differ_n/differ_d + mu[,j]
    }
    return(expected_y)
  }
  
  ##------------------------------------------------------##
  ##-------- CR y_residual_square --------------##
  ##------------------------------------------------------##
  # e.Norm.mix.cr.bb <- function(c,cx,cn,g, betas, j,sigma2)
  # {
  #   sigma = sqrt(sigma2)
  #   expected_y = matrix(0,cn,g)
  #   mu = cx %*% betas
  #   # for(j in 1:g){
  #   #   # 期望y的分子和分母
  #   #   residual_l <- c[,1] - mu[,j]
  #   #   residual_u <- c[,1] - mu[,j]
  #   #   differ_n <- residual_l*dnorm(c[,1],mu[,j],sigma[j]) - 
  #   #     residual_u*dnorm(c[,2],mu[,j],sigma[j]) 
  #   #   differ_d <- pnorm(c[,2],mu[,j],sigma[j]) - dnorm(c[,1],mu[,j],sigma[j])
  #   #   expected_y[,j] <- differ_n/differ_d + + sigma2[j]
  #   # }
  #   return(expected_y)
  # }
  
  
  start.time = Sys.time()
  
  mu = x %*% betas
  cmu <- cx %*% betas #　期望残差
  # 得加上cdf，否则直接停止;还有求和
  lk = lk.old = sum(d.Norm.mix.reg(y, mu, sigma2, pI, log = T))+
    sum(p.Norm.mix.reg(c, cmu, sigma2, pI, log = T))
  
  criterio  <- 1 # 判别准则
  count  <- 0
  if(isTRUE(print)){
    cat(paste(rep("-", 70), sep = "", collapse = ""), "\n")
    cat("iter =", count, "\t logli.old=",  lk.old, "\n")
    cat(paste(rep("-", 70), sep = "", collapse = ""), "\n")
  }
  
  repeat
  {
    count = count + 1
    tal = matrix(0, n, g)  # Z矩阵
    talc =  matrix(0,cn, g) # 删失的Z
    sigma_values <- sqrt(sigma2)
    for(j in 1:g){
      tal[, j] = pI[j] * dnorm(y, mu[, j], sigma_values[j])
      talc[, j] = pI[j] * (pnorm(c[,2], cmu[, j],sigma_values[j])-
                             pnorm(c[,1], cmu[, j], sigma_values[j]))
    }
    
    for(k in 1:n) if(all(tal[k,] == 0)) tal[k,] = .Machine$double.xmin
    for(k in 1:cn) if(all(talc[k,] == 0)) talc[k,] = .Machine$double.xmin
    tal = tal/rowSums(tal)  # 应该是Z矩阵
    talc = talc/rowSums(talc)
    y_expected <- e.Norm.mix.cr.y(c,cx,cn,g, betas, sigma2) # 计算期望y，用于beta

    SS = 0
    for (j in 1:g)
    {
      ### M-step: 
      temp_tal <- c(tal[, j],talc[,j])
      temp_y <- c(y,y_expected[,j])
      temp_x <- rbind(x,cx)
      pI[j] = sum(temp_tal) / (n+cn) 
      Sxy = t(temp_x) %*% diag(temp_tal) %*% temp_y   # t(x) 函数用于对矩阵或数据框进行转置
      Sxx = t(temp_x) %*% diag(temp_tal) %*% temp_x
      # betas[, j] = inv.mat(Sxx) %*% Sxy  # 这是求逆吗？
      betas[, j] = solve(Sxx) %*% Sxy
      
      mu[, j] = x %*% betas[, j]
      
      sigma_values = sqrt(sigma2)
      cmu[,j] <- cx %*% betas[,j]
      # 期望y的分子和分母
      residual_l <- c[,1] - cmu[,j]
      residual_u <- c[,1] - cmu[,j]
      differ_n <- sigma2[j]*(residual_l*dnorm(c[,1],cmu[,j],sigma_values[j]) - residual_u*dnorm(c[,2],cmu[,j],sigma_values[j]))
      differ_d <- pnorm(c[,2],cmu[,j],sigma_values[j]) - dnorm(c[,1],cmu[,j],sigma_values[j])
      cbb <- differ_n/differ_d + sigma2[j]  # 计算期望残差平方
    
      
      bb = (y - mu[, j])^2   # 残差平方
      sigma2[j] = sum(temp_tal * c(bb,cbb)) / sum(temp_tal)
      if(fix.sigma) SS = sum(tal[, j] * bb) + SS
    }
    
    if(fix.sigma) sigma2 = rep(SS/n, g)
    
    lk.new = sum(d.Norm.mix.reg(y, mu, sigma2, pI, log = T)) +
      sum(p.Norm.mix.reg(c, cmu, sigma2, pI, log = T))
    
    if(is.nan(lk.new)) {
      lk.new = lk.old
      break
    }
    
    lk = c(lk, lk.new)
    if(Stp.rule == "Log.like") criterio = (lk.new - lk.old)/abs(lk.old)
    else{ criterio = Stop.rule(lk) }
    
    diff = lk.new -lk.old
    if(count %% per == 0 || is.na(diff))
    {
      if(isTRUE(print)){
        cat('iter =', count, '\t logli =', lk.new, '\t diff =', 
            diff, Stp.rule, "'s diff =", criterio, '\n')
        cat(paste(rep("-", 60), sep = "", collapse = ""), "\n")
      }
    }
    
    if(criterio < error | count == iter.max) break
    lk.old = lk.new
  }
  
  # End of the estimation process
  lk         <- lk.new
  end.time        <- Sys.time()
  time.taken      <- end.time - start.time
  
  
  m <- g * (p + 1) + (g - 1) # Abeta + Sigma + alpha
  if(fix.sigma) m = m - g + 1
  
  aic = -2 * lk + 2 * m
  bic = -2 * lk + log(n) * m
  edc = -2 * lk + 0.2 * sqrt(n) * m
  aic_c = -2 * lk + 2 * n * m / (n - m - 1)
  abic  = -2 * lk + m * log((n + 2) / 24)
  
  Group = apply(tal, 1, which.max)
  
  if(!is.null(Class)){
    true.clus = Class
    km.clus = Group
    tab = table(true.clus, km.clus)
    MCR = 1 - sum(diag(tab))/sum(tab)
    RII = aricode :: clustComp(km.clus, true.clus) # 评价指标库
    
    obj.out = list( time = time.taken, group = Group, m = m, betas = betas,
                    sigma2 = sigma2, pI = pI, loglike = lk, aic = aic,
                    bic = bic, edc = edc, aic_c = aic_c, abic = abic,
                    iter = count, MCR = MCR, similarity = t(as.matrix(RII)), 
                    cross_class = tab, Zij = tal, 
                    convergence = criterio < error, crite = criterio)
  }
  else{obj.out <-
    list( time = time.taken, group = Group, m = m, betas = betas,
          sigma2 = sigma2, pI = pI, loglike = lk, aic = aic, 
          bic = bic, edc = edc, aic_c = aic_c, abic = abic,
          iter = count, convergence = criterio < error, 
          crite = criterio)} 
  obj.out
}

