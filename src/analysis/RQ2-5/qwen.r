library(jsonlite)
library(dplyr)
library(tidyr)
library(ggplot2)
library(rstatix)
library(FSA)
library(effsize)

# === Set output directory ===
output_dir <- "/Users/leehanyun/Desktop/logs"

# === Load CSV of F1/ROUGE metrics ===
metrics_csv <- read.csv("/Users/leehanyun/Desktop/accuracy_summary.csv", stringsAsFactors = FALSE)

# === Collect all run_ids to remove ===
# Suppose your outlier rows are already in a variable `removed_df_all` (from previous IQR)
# Here, we collect run_ids from CSV for filtering
run_ids_to_remove <- unlist(strsplit(metrics_csv$run_ids, ",\\s*"))
run_ids_to_remove <- trimws(run_ids_to_remove)  # remove any leading/trailing spaces
run_ids_to_remove <- unique(run_ids_to_remove)

# === Define IQR-based outlier removal ===
remove_outliers_iqr <- function(df, col_name) {
  Q1 <- quantile(df[[col_name]], 0.25, na.rm = TRUE)
  Q3 <- quantile(df[[col_name]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  mask <- df[[col_name]] >= (Q1 - 1.5*IQR) & df[[col_name]] <= (Q3 + 1.5*IQR)
  removed <- df[!mask, ]
  df_clean <- df[mask, ]
  return(list(clean=df_clean, removed=removed))
}

# === Load JSON logs ===
load_logs <- function(folder) {
  files <- list.files(folder, pattern="\\.log$", full.names = TRUE)
  data <- lapply(files, function(f) {
    log <- tryCatch(fromJSON(f), error=function(e) NULL)
    if (!is.null(log) && !is.null(log$tokens) && !is.null(log$inference_seconds)) {
      data.frame(
        file_name = basename(f),
        latency_s_per_token = log$inference_seconds / log$tokens,
        memory_mb = log$max_rss_children_kb / 1024,
        cpu_percent = log$avg_cpu_percent,
        energy_j_per_token = log$energy_j / log$tokens,
        run_id = gsub("\\.log$","",basename(f)), # Add run_id column
        stringsAsFactors = FALSE
      )
    } else {
      NULL
    }
  })
  bind_rows(data)
}

# === Quantization folders ===
folders <- list(
  FP16 = file.path(output_dir, "qwen-3b-fp16_logs"),
  INT8 = file.path(output_dir, "qwen-3b-int8_logs"),
  INT4 = file.path(output_dir, "qwen-3b-int4_logs")
)

# === Load & clean data ===
results <- list()
removed_logs <- list()
metrics <- c("latency_s_per_token","memory_mb","cpu_percent","energy_j_per_token")

for(name in names(folders)){
  df <- load_logs(folders[[name]])
  cat("\n=== ", name, " ===\nLoaded ", nrow(df), " logs\n")
  
  # --- Remove logs whose run_id is in CSV outliers ---
  csv_outlier_mask <- df$run_id %in% run_ids_to_remove
  if(any(csv_outlier_mask)){
    removed_csv <- df[csv_outlier_mask, ]
    removed_csv$reason <- "csv_outlier"
    removed_csv$model <- name
    removed_logs[[paste0(name,"_csv")]] <- removed_csv
    df <- df[!csv_outlier_mask, ]
    cat("Removed ", nrow(removed_csv), " CSV outlier logs\n")
  }

  # --- Remove IQR outliers ---
  clean_df <- df
  removed_total <- data.frame()
  for(m in metrics){
    out <- remove_outliers_iqr(clean_df, m)
    clean_df <- out$clean
    if(nrow(out$removed)>0){
      out$removed$reason <- paste0(m,"_outlier")
      removed_total <- bind_rows(removed_total, out$removed)
    }
  }
  
  if(nrow(removed_total) >0){
    removed_total$model <- name
    removed_logs[[name]] <- removed_total
  }
  
  results[[name]] <- clean_df
  cat("After removing outliers: ", nrow(clean_df), " rows kept, ", 
      nrow(removed_total) + sum(csv_outlier_mask), " removed\n")
}

# Save removed outliers
if(length(removed_logs)>0){
  removed_df_all <- bind_rows(removed_logs)
  write.csv(removed_df_all[,c("model","file_name","run_id","reason")],
            file=file.path(output_dir,"qwen_removed_outliers.csv"), row.names=FALSE)
  cat("⚠️ Saved removed outliers\n")
} else {cat("No outliers removed\n")}

# === Summary statistics ===
# === Extended Summary Statistics ===
summary_list <- list()

for(name in names(results)){
  df <- results[[name]]
  
  # Compute mean, SD, median, min, max, variance for each metric
  stats <- lapply(metrics, function(m){
    c(
      mean = mean(df[[m]], na.rm=TRUE),
      sd = sd(df[[m]], na.rm=TRUE),
      median = median(df[[m]], na.rm=TRUE),
      min = min(df[[m]], na.rm=TRUE),
      max = max(df[[m]], na.rm=TRUE),
      var = var(df[[m]], na.rm=TRUE)
    )
  })
  
  # Convert to data.frame
  stats_df <- as.data.frame(do.call(rbind, stats))
  stats_df$Metric <- metrics
  stats_df$Quantization <- name
  rownames(stats_df) <- NULL
  
  summary_list[[name]] <- stats_df
}

# Combine all quantizations
final_summary <- bind_rows(summary_list) %>%
  select(Metric, Quantization, mean, sd, median, min, max, var)

# Save extended summary
write.csv(final_summary,
          file=file.path(output_dir, "qwen_quantization_extended_summary.csv"),
          row.names=FALSE)

cat("Saved extended summary statistics (mean, SD, median, min, max, variance)\n")


# === Prepare data for plotting & stats ===
plot_data <- bind_rows(lapply(names(results), function(name){
  df <- results[[name]]
  df$model <- name
  pivot_longer(df, all_of(metrics), names_to="metric", values_to="value")
}))

# Convert model to character to avoid factor issues
plot_data$model <- as.character(plot_data$model)

# === Boxplot 2x2 ===
p_box <- ggplot(plot_data, aes(x=model, y=value, fill=model)) +
  geom_boxplot(outlier.colour="red", outlier.shape=16) +
  facet_wrap(~metric, scales="free", ncol=2) +
  theme_bw() +
  labs(title="Boxplots of Metrics by Quantization Level (Qwen)") +
  scale_fill_manual(values=c("FP16"="blue","INT8"="orange","INT4"="green"))
ggsave(filename=file.path(output_dir,"qwen_all_metrics_boxplot.png"), plot=p_box,
       dpi=300, width=12, height=8)

# === Density plot 2x2 ===
p_dens <- ggplot(plot_data, aes(x=value, fill=model)) +
  geom_density(alpha=0.3) +
  facet_wrap(~metric, scales="free", ncol=2) +
  theme_bw() +
  labs(title="Density Plots of Metrics by Quantization Level (Qwen)") +
  scale_fill_manual(values=c("FP16"="blue","INT8"="orange","INT4"="green"))
ggsave(filename=file.path(output_dir,"qwen_all_metrics_density.png"), plot=p_dens,
       dpi=300, width=12, height=8)

# === Statistical tests with transformation and automatic parametric/non-parametric selection (safe) ===
stat_results_all <- data.frame(
  Metric = character(),
  Test = character(),
  Group1 = character(),
  Group2 = character(),
  Statistic = numeric(),
  P_value = numeric(),
  Effect_Size = numeric(),
  stringsAsFactors = FALSE
)

is_constant <- function(x) length(unique(x)) == 1  # 判斷是否全一樣

for(m in metrics){
  df_metric <- plot_data %>% filter(metric == m)
  
  # Initialize transformed column
  df_metric$value_trans <- df_metric$value
  
  # --- Initial normality check with safety ---
  sw_results <- df_metric %>% group_by(model) %>%
    summarise(shapiro_p = if(is_constant(value_trans)) NA else shapiro.test(value_trans)$p.value,
              .groups="drop")
  
  # --- Apply transformation if any group is non-normal ---
  if(any(!is.na(sw_results$shapiro_p) & sw_results$shapiro_p <= 0.05)){
    if(all(df_metric$value_trans > 0)){
      df_metric$value_trans <- log(df_metric$value_trans)
      cat("Log transformation applied for metric:", m, "\n")
    } else if(all(df_metric$value_trans >= 0)){
      df_metric$value_trans <- sqrt(df_metric$value_trans)
      cat("Sqrt transformation applied for metric:", m, "\n")
    } else {
      min_val <- min(df_metric$value_trans, na.rm=TRUE)
      df_metric$value_trans <- sqrt(df_metric$value_trans - min_val + 1)
      cat("Power transformation applied for metric:", m, "\n")
    }
  }
  
  # --- Re-check normality after transformation ---
  sw_results_post <- df_metric %>% group_by(model) %>%
    summarise(shapiro_p = if(is_constant(value_trans)) NA else shapiro.test(value_trans)$p.value,
              .groups="drop")
  
  # --- Decide parametric or non-parametric based on post-transformation normality ---
  if(all(is.na(sw_results_post$shapiro_p) | sw_results_post$shapiro_p > 0.05)){
    # Parametric: ANOVA + pairwise t-test
    res_aov <- aov(value_trans ~ model, data=df_metric)
    print(summary(res_aov))
    
    posthoc <- pairwise.t.test(df_metric$value_trans, df_metric$model, p.adjust.method="bonferroni")
    
    groups <- unique(df_metric$model)
    for(i in 1:(length(groups)-1)){
      for(j in (i+1):length(groups)){
        g1 <- groups[i]
        g2 <- groups[j]
        # Safe p-value extraction
        if(g1 %in% rownames(posthoc$p.value) && g2 %in% colnames(posthoc$p.value)){
          p_val <- posthoc$p.value[g1, g2]
        } else if(g2 %in% rownames(posthoc$p.value) && g1 %in% colnames(posthoc$p.value)){
          p_val <- posthoc$p.value[g2, g1]
        } else {
          p_val <- NA
        }
        # Cohen's d
        d <- cohen.d(df_metric$value_trans[df_metric$model==g1],
                     df_metric$value_trans[df_metric$model==g2])$estimate
        stat_results_all <- rbind(stat_results_all, data.frame(
          Metric = m,
          Test = "t-test (post-hoc)",
          Group1 = g1,
          Group2 = g2,
          Statistic = NA,
          P_value = p_val,
          Effect_Size = d
        ))
      }
    }
    
  } else {
    # Non-parametric: Kruskal-Wallis + Dunn
    res_kw <- kruskal.test(value_trans ~ model, data=df_metric)
    print(res_kw)
    
    posthoc <- dunnTest(value_trans ~ model, data=df_metric, method="bonferroni")
    dunn_pairs <- posthoc$res
    
    for(r in 1:nrow(dunn_pairs)){
      groups_split <- strsplit(as.character(dunn_pairs$Comparison[r]), " - ")[[1]]
      g1 <- groups_split[1]
      g2 <- groups_split[2]
      delta <- cliff.delta(df_metric$value_trans[df_metric$model==g1],
                          df_metric$value_trans[df_metric$model==g2])$estimate
      stat_results_all <- rbind(stat_results_all, data.frame(
        Metric = m,
        Test = "Dunn post-hoc",
        Group1 = g1,
        Group2 = g2,
        Statistic = NA,
        P_value = dunn_pairs$P.adj[r],
        Effect_Size = delta
      ))
    }
  }
}

# Save results
write.csv(stat_results_all, file=file.path(output_dir,"qwen_statistical_tests_final_safe.csv"), row.names=FALSE)
cat("Saved all statistical test results (parametric/non-parametric) with effect sizes\n")
