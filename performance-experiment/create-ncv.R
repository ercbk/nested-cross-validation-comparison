# nested-cv data function




create_ncv <- function(dat, repeats, method) {
   
   attempt::stop_if_not(repeats, is.numeric, "repeats needs to be a numeric class")
   attempt::stop_if_not(method, is.character, "method needs to be a character class")
   
   grid <- tidyr::crossing(dat, repeats)
   
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




