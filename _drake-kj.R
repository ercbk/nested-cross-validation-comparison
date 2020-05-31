# drake make file for Kuhn-Johnson performance experiment


# Notes:
# 1. see plan-kj.R for more details on how this thing works
# 2. link to {future} issue with instructions on special PuTTY settings, https://github.com/HenrikBengtsson/future/issues/370


# load packages, functions, and drake plan
source("performance-experiment/packages.R")
source("performance-experiment/functions/mlbench-data.R")
source("performance-experiment/functions/create-ncv-objects.R")
source("performance-experiment/functions/create-models.R")
source("performance-experiment/functions/create-grids.R")
source("performance-experiment/functions/inner-tune.R")
source("performance-experiment/functions/outer-cv.R")
source("performance-experiment/functions/ncv-compare.R")
source("performance-experiment/functions/run-ncv.R")
source("performance-experiment/Kuhn-Johnson/plan-kj.R")




set.seed(2019)

# Using different compute sizes for each model
ip1 <- Sys.getenv("GLMEC2IP")
ip2 <- Sys.getenv("RFEC2IP")
public_ips <- c(ip1, ip2)
# ppk file converted by PuTTY from an AWS pem file
ssh_private_key_file <- Sys.getenv("AWSKEYPATH")


cl <- future::makeClusterPSOCK(
      
      ## Public IP numbers of EC2 instances
      public_ips,
      
      ## User name (always 'ubuntu')
      user = "ubuntu",
      
      ## Use private SSH key registered with AWS
      ## futureSettings is a saved PuTTY session with settings to keep ssh active
      rshcmd = c("plink", "-ssh", "-load", "futureSettings","-i", ssh_private_key_file),
      rshopts = c(
            "-sshrawlog", "ec2-ssh-raw.log"
      ),
      
      rscript_args = c("-e", shQuote(".libPaths('/home/rstudio/R/x86_64-pc-linux-gnu-library/3.6')")
      ), 
      verbose = TRUE
)


future::plan(list(tweak(cluster, workers = cl), multiprocess))

cache_kj <- drake_cache(path = ".drake")

# verbose = 0 prints nothing, verbose = 1 prints message as each target completes; verbose = 2 adds a progress bar that tracks target completion
drake_config(
      plan,
      verbose = 1,
      lock_envir = FALSE,
      jobs_preprocess = 7,
      cache = cache_kj
)

