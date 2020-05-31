# r_make
# runs experiment in a clean environment in a separate r session


# Notes
# 1. Didn't make the render readme function into a target because the readd doesn't explicitly call a specific target, so drake isn't triggered to build it. The buildtimes function in the readme also doesn't trigger a build.


# text me if an error occurs
options(error = function() {
      library(RPushbullet)
      pbPost("note", "Error", geterrmessage())
      if(!interactive()) stop(geterrmessage())
})



drake::r_make(source = "_drake-raschka.R")

rmarkdown::render(
      input = "README.Rmd"
)


# text me when it finishes
RPushbullet::pbPost("note", title="raschka performance experiment", body="ncv run finished")

