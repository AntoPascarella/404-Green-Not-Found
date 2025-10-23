library(ggplot2)
library(dplyr)
library(patchwork)

# ---- Load data ----
df <- read.csv("results_accuracy/accuracy_clean.csv")

# Check the model names (to match filter exactly)
unique(df$model)

# ---- Filter only LLaMA-2-3B ----
df_llama <- df %>% filter(grepl("llama", tolower(model)))

pal <- c("FP16"="#4C72B0","INT8"="#55A868","INT4"="#C44E52")

# ADDED Data
# === Clean data before plotting (remove non-finite / irrelevant rows) ===
df_f1 <- df %>%
  filter(task_type == "QA", is.finite(f1), !is.na(quantization_level))
df_roug <- df %>%
  filter(task_type %in% c("SUMM_SHORT","SUMM_LONG"),
         is.finite(rougeL), !is.na(quantization_level))

# Optional: clamp to [0,1] in case of rounding or outlier logs
df_f1$f1       <- pmin(pmax(df_f1$f1, 0), 1)
df_roug$rougeL <- pmin(pmax(df_roug$rougeL, 0), 1)

# Merge for combined plotting if your plots expect a unified dataframe
plot_data <- bind_rows(
  df_f1 %>% mutate(metric = "F1", value = f1),
  df_roug %>% mutate(metric = "ROUGE-L", value = rougeL)
)


# ============ BOX PLOTS ============

p_f1_box <- ggplot(df_llama, aes(x = quantization_level, y = f1, fill = quantization_level)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, alpha = 0.85) +
  scale_fill_manual(values = pal) +
  theme_bw() +
  labs(title = "(a) LLaMA-2-3B: F1 distribution by quantization",
       x = "Quantization level", y = "F1")

p_rouge_box <- ggplot(df_llama, aes(x = quantization_level, y = rougeL, fill = quantization_level)) +
  geom_boxplot(outlier.colour = "red", outlier.shape = 16, alpha = 0.85) +
  scale_fill_manual(values = pal) +
  theme_bw() +
  labs(title = "(b) LLaMA-2-3B: ROUGE-L distribution by quantization",
       x = "Quantization level", y = "ROUGE-L")

p_box_llama <- p_f1_box / p_rouge_box
ggsave("fig_accuracy_boxplots_llama.png", p_box_llama, width = 8, height = 7, dpi = 300)

# ============ DENSITY PLOTS ============

p_f1_dens <- ggplot(df_llama, aes(x = f1, fill = quantization_level)) +
  geom_density(alpha = 0.4) +
  scale_fill_manual(values = pal) +
  theme_bw() +
  labs(title = "(c) LLaMA-2-3B: F1 density by quantization", x = "F1", y = "Density")

p_rouge_dens <- ggplot(df_llama, aes(x = rougeL, fill = quantization_level)) +
  geom_density(alpha = 0.4) +
  scale_fill_manual(values = pal) +
  theme_bw() +
  labs(title = "(d) LLaMA-2-3B: ROUGE-L density by quantization", x = "ROUGE-L", y = "Density")

p_dens_llama <- p_f1_dens / p_rouge_dens
ggsave("fig_accuracy_density_llama.png", p_dens_llama, width = 8, height = 7, dpi = 300)