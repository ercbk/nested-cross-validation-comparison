# nested-cv data function

# inputs:
# 1. dat = dataset
# 2. repeats = numeric vector with numbers of repeats
# 3. method = "kj" or "raschka"
# outputs:
# 1. list of {rsample} nested cv objects; one object per repeat value



create_ncv_objects <- function(dat, repeats, method) {
   
   attempt::stop_if_not(repeats, is.numeric, "repeats needs to be a numeric class")
   attempt::stop_if_not(method, is.character, "method needs to be a character class")
   
   # don't remember but guessing crossing needs a list object
   if (is.data.frame(dat)) {
      dat <- list(dat)
   }
   # tibble grid of data and repeats
   grid <- tidyr::crossing(dat, repeats)
   
   # generate list of ncv objects
   # dynGet needed to get reps out of the envirnonment and into the nested_cv function
   if (method == "kj") {
      ncv_list <- purrr::map2(grid$dat, grid$repeats, function(dat, reps) {
         rsample::nested_cv(dat,
                            outside = vfold_cv(v = 10, repeats = dynGet("reps")),
                            inside = bootstraps(times = 25))
      })
   } else if (method == "raschka") {
      ncv_list <- purrr::map2(grid$dat, grid$repeats, function(dat, reps) {
         rsample::nested_cv(dat,
                            outside = vfold_cv(v = 5, repeats = dynGet("reps")),
                            inside = vfold_cv(v = 2))
      })
   } else {
      stop("Need to specify method as kj or raschka", call. = FALSE)
   }
   
   return(ncv_list)
}




