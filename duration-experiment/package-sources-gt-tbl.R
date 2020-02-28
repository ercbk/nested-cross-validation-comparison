# GT table: Package Sources for the Model Functions


pacman::p_load(tibble, dplyr, tidyr, gt)

runs_raw <- readr::read_rds("data/duration-runs.rds") %>%
      mutate(implementation = stringr::str_to_title(implementation))


# hardcoded
# packages used for the model functions ordered by implementation in the runs_raw file
elast_net <- c("sklearn", "sklearn", "parsnip-glmnet", "mlr-glmnet", "parsnip-glmnet", "h2o", "parsnip-glmnet", "parsnip-glmnet", "parsnip-glmnet")
rand_forest <- c("sklearn", "sklearn", "ranger", "mlr-ranger", "parsnip-ranger", "h2o", "sklearn", "parsnip-ranger", "ranger")


elast_dat <- runs_raw %>% 
      select(implementation) %>% 
      bind_cols(`Elastic Net` = elast_net) %>% 
      pivot_wider(names_from = implementation, values_from = `Elastic Net`)

# implementations as cols, algorithm as rows, values = package used
model_dat <- runs_raw %>% 
      select(implementation) %>% 
      bind_cols(`Random Forest` = rand_forest) %>% 
      pivot_wider(names_from = implementation, values_from = `Random Forest`) %>% 
      bind_rows(elast_dat) %>% 
      mutate(rowname = c("Random Forest", "Elastic Net"))

model_dat %>% 
      gt() %>% 
      tab_spanner(
            label = "Implementation",
            columns = everything()
      ) %>% 
      data_color(columns = vars(Reticulate, Python, Mlr3, `Ranger-Kj`),
                 colors = scales::col_factor(
                       palette = "#195198",
                       domain = c("sklearn", "ranger", "parsnip-glmnet", "mlr-ranger", "mlr-glmnet" ))) %>% 
      data_color(columns = vars(Tune, H2o, Sklearn, Parsnip, Ranger),
                 colors = scales::col_factor(
                       palette = "#BD9865",
                       domain = c("sklearn", "ranger", "parsnip-glmnet", "parsnip-ranger", "h2o" ))) %>%
      tab_style(
            style = cell_text(align = "center"),
            locations = cells_body()
      ) %>% 
      tab_options(
            table.background.color = "ivory",
            table.border.top.style = "None"
      ) %>% 
      tab_header(
            title = "Package Sources for the Model Functions"
      ) %>% 
      gtsave(filename = "duration-pkg-tbl.png",
             path = "duration-experiment/outputs")
