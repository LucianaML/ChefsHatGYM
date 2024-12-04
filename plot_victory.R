library(jsonlite)
library(openxlsx)
library(ggplot2)
library(dplyr)


df_count_score <- function(data, dir_path, test, value) {
  
  df_summary <- data %>%
    filter(Score == value) %>%  # Filtra apenas as linhas onde o score é 3
    group_by(Player) %>%  # Agrupa por jogador
    summarise(winning = n(), .groups = "drop") %>%  # Conta o número de 3 pontos por jogador e partida
    ungroup() 
  
  plot_test <- ggplot(df_summary, aes(x = Player, y = winning, fill = Player)) +
    geom_bar(stat = "identity") +
    labs(title = "Pontuação Total por Jogador", x = "Jogador", y = "vitorias") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  
  ggsave(paste0(dir_path, "/grafico_test_value-", value, ".png"), 
         plot = plot_test, width = 8, height = 6, units = "in", dpi = 300) 
  
  #return(df_summary)
}

path <- paste0(getwd(), '/trained/plot_by_R/')

test_datasets <- c('22-1509_train_vsRandom.csv')

df_test <- data.frame()

for(i in 1:length(test_datasets)){
  dir_path <- paste0(path, "/plots_test_vsRandom_vsDQL")
  
  file_name <- paste0(test_datasets[1])
  df <- read.csv(paste0(getwd(), '/trained/', file_name))
  
  df_filtred <- df
  
  df_data <- subset(df_filtred, select = c('Match.Number'))
  df_match_score <- df_filtred['Current.Score']
  
  score_match <- lapply(1:nrow(df_match_score), function(i) {
    data.frame(elemento = df_match_score$Current.Score[[i]])
  })
  
  df_score <- data.frame()
  
  for(data in  1:length(score_match)){
      str <- as.character(score_match[[data]])
      score_list <- strsplit(str, ",")[[1]]
      
      player_names <- gsub("'", "", sapply(score_list, function(x) strsplit(x, ":")[[1]][1]))  
      scores <- as.numeric(sapply(score_list, function(x) strsplit(x, ":")[[1]][2]))
      df_temp <- data.frame(Player = player_names, Score = scores)
      
      df_temp$Match <- data
      
      
      if (nrow(df_score) == 0) {
        df_score <- df_temp
      } else {
        df_score <- rbind(df_score, df_temp)
      }
      
      plot_ <- ggplot(df_temp, aes(x = Player, y = Score, fill = Player)) +
        geom_bar(stat = "identity") +
        labs(title = "Pontuação Total por Jogador", x = "Jogador", y = "Pontuação Total") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
      
      if(!dir.exists(dir_path)){
        dir.create(dir_path)
      }
      
      ggsave(paste0(dir_path, "/grafico_", data, ".png"), 
             plot = plot_,  # Usa o último gráfico gerado
             width = 8, height = 6, units = "in", dpi = 300) 
  }
  
  df_count_score(df_score, dir_path, i, 0)
  df_count_score(df_score, dir_path, i, 1)
  df_count_score(df_score, dir_path, i, 2)
  df_count_score(df_score, dir_path, i, 3)
  
  #colnames(df_score) <- names(data_list)
  
  #df_final <- cbind(df_data, df_score)
  
  #saveRDS(df_final ,paste0(path, '/test_', i, '.RDS'))
  write.xlsx(df_score, paste0(path, '/test_vs_Random', i, '.csv'))
  df_test <- rbind(df_test, df_score)
}





