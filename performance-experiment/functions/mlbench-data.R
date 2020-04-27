# create simulation data

# Inputs are 10 independent variables uniformly distributed on the interval [0,1], only 5 out of these 10 are actually used. Outputs are created according to the formula
# y = 10 sin(π x1 x2) + 20 (x3 - 0.5)^2 + 10 x4 + 5 x5 + e

mlbench_data <- function(n, noise_sd = 1, seed = 2019) {
      set.seed(seed)
      tmp <- mlbench::mlbench.friedman1(n, sd = noise_sd)
      tmp <- cbind(tmp$x, tmp$y)
      tmp <- as.data.frame(tmp)
      names(tmp)[ncol(tmp)] <- "y"
      tmp
}