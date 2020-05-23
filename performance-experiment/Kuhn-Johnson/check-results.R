# Results


library(drake); library(dplyr)

loadd(ncv_results_100)
View(ncv_results_100)
loadd(ncv_results_800)
View(ncv_results_800)
loadd(ncv_results_2000)
View(ncv_results_2000)
loadd(ncv_results_5000)
View(ncv_results_5000)

# each target's build time
bt <- build_times(starts_with("ncv_results"), digits = 4)
View(bt)
bt %>% 
      select(target, elapsed) %>% 
      kableExtra::kable() %>% 
      kableExtra::save_kable(file = "performance-experiment/output/kj-build-times.png")
