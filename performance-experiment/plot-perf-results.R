pacman::p_load(dplyr, ggplot2)

dat_line <- perf_results_100 %>% 
      select(n = `100`, repeats, delta_error) %>% 
      mutate(method = "kj")

g <- ggplot(dat_line, aes(x = repeats, y = delta_error, group = n)) +
      geom_point(aes(color = n)) +
      geom_line(aes(color = n)) +
      scale_y_continuous(limits = c(0, 0.3))

g

dat_bar <- dat_line %>% 
      group_by(n) %>%
      filter(delta_error == min(delta_error))

p <- ggplot(df, aes(x = n, y = delta_error)) +
   geom_bar(
      aes(color = method, fill = method),
      stat = "identity", position = position_dodge(0.8),
      width = 0.7
   ) +
   scale_color_manual(values = c("#0073C2FF", "#EFC000FF"))+
   scale_fill_manual(values = c("#0073C2FF", "#EFC000FF"))
p

p + geom_text(
   aes(label = delta_error, group = method), 
   position = position_dodge(0.8),
   vjust = -0.3, size = 3.5
)

