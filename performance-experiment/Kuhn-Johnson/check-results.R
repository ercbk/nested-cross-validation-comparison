# Results


library(drake); library(dplyr)

# loadd(perf_results_100)
# View(perf_results_100)
loadd(perf_results_800)
View(perf_results_800)

# each target's build time
bt <- build_times(starts_with("ncv_results"), digits = 4)
View(bt)
bt %>% 
      select(target, elapsed) %>% 
      kableExtra::kable() %>% 
      kableExtra::save_kable(file = "performance-experiment/output/kj-build-times.png")
