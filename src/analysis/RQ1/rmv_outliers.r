library(dplyr)
library(readr)
library(rstatix)
library(FSA)      # Added: Required for dunnTest()
library(effsize)  # Added: Required for cliff.delta()

# === Config ===
input_csv  <- "runplan_results.csv"   # has: model, quantization_level, task_type, f1, rougeL, ...
output_dir <- "results_accuracy"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# === Load data ===
df <- read_csv(input_csv)

# === IQR-based outlier removal (GLOBAL) ===
# [Note: Removed the unused 'remove_outliers_iqr' function]

remove_outliers_iqr_vec <- function(v) {
  v0 <- v[!is.na(v)]
  if (length(v0) < 4) return(list(mask = rep(TRUE, length(v))))  # too few points
  Q1 <- quantile(v0, 0.25, na.rm=TRUE)
  Q3 <- quantile(v0, 0.75, na.rm=TRUE)
  IQRv <- Q3 - Q1
  if (!is.finite(IQRv) || IQRv == 0) {
    # Degenerate distribution: skip filtering
    return(list(mask = rep(TRUE, length(v))))
  }
  lower <- Q1 - 1.5*IQRv
  upper <- Q3 + 1.5*IQRv
  # Start with "keep"
  mask <- rep(TRUE, length(v))
  # Only evaluate non-NA entries
  idx <- which(!is.na(v))
  mask[idx] <- v[idx] >= lower & v[idx] <= upper
  list(mask = mask, lower = lower, upper = upper)
}

metrics <- c("f1","rougeL")
removed_all <- list()
clean_df <- df

for (m in metrics) {
  if (!m %in% names(clean_df)) next

  # Task-aware subset to avoid nuking NAs:
  if (m == "f1" && "task_type" %in% names(clean_df)) {
    sub_idx <- which(clean_df$task_type == "QA" & !is.na(clean_df[[m]]))
  } else if (m == "rougeL" && "task_type" %in% names(clean_df)) {
    sub_idx <- which(clean_df$task_type %in% c("SUMM_SHORT","SUMM_LONG") & !is.na(clean_df[[m]]))
  } else {
    sub_idx <- which(!is.na(clean_df[[m]]))
  }

  if (length(sub_idx) < 4) next  # nothing to do

  # Build a full-length mask defaulting to TRUE
  full_mask <- rep(TRUE, nrow(clean_df))

  # Compute IQR mask only on the subset
  res <- remove_outliers_iqr_vec(clean_df[[m]][sub_idx])
  full_mask[sub_idx] <- res$mask

  # Collect removed rows (reasoned by metric)
  rem <- clean_df[!full_mask, , drop = FALSE]
  if (nrow(rem) > 0) {
    rem$reason <- paste0(m, "_outlier_global")
    removed_all[[length(removed_all) + 1]] <- rem
  }

  # Keep only rows passing this metric’s filter
  clean_df <- clean_df[full_mask, , drop = FALSE]
}


removed_df <- if (length(removed_all) > 0) bind_rows(removed_all) else tibble()

# === Save results ===
write_csv(clean_df, file.path(output_dir, "accuracy_clean.csv"))
if (nrow(removed_df) > 0) {
  write_csv(removed_df, file.path(output_dir, "accuracy_outliers_removed.csv"))
  cat("⚠️ Outliers removed globally → results_accuracy/accuracy_outliers_removed.csv\n")
} else {
  cat("✅ No global outliers detected\n")
}

# === Optional: quick summary after removal ===
if ("run_id" %in% names(clean_df)) {
  summary_stats <- clean_df %>%
    group_by(model, quantization_level) %>%
    summarise(
      mean_f1 = mean(f1, na.rm = TRUE),
      sd_f1   = sd(f1, na.rm = TRUE),
      mean_rouge = mean(rougeL, na.rm = TRUE),
      sd_rouge   = sd(rougeL, na.rm = TRUE),
      n = n(),
      run_ids = paste(run_id, collapse = ", "),
      .groups = "drop"
    )
  write_csv(summary_stats, file.path(output_dir, "accuracy_summary.csv"))
  cat("✅ Saved summary stats → results_accuracy/accuracy_summary.csv\n")
} else {
  cat("ℹ️ Skipping summary stats (no 'run_id' column found).\n")
}


# Shapiro–Wilk for QA (F1)
sw_f1 <- df %>%
  filter(task_type=="QA", !is.na(f1)) %>%
  group_by(quantization_level) %>%
  summarise(
    # Add check for variance. sd() is NA if n<2, 0 if constant.
    is_testable = n() >= 3 && !is.na(sd(f1)) && sd(f1) > 0,
    W = if(is_testable) shapiro_test(f1)$statistic else NA_real_,
    p = if(is_testable) shapiro_test(f1)$p.value else NA_real_,
    n = n(),
    .groups="drop"
  ) %>% 
  mutate(metric="F1") %>%
  select(-is_testable) # Clean up helper column

# Shapiro–Wilk for SUMM (ROUGE-L)
sw_rouge <- df %>%
  filter(task_type %in% c("SUMM_SHORT","SUMM_LONG"), !is.na(rougeL)) %>%
  group_by(quantization_level) %>%
  summarise(
    # Add check for variance.
    is_testable = n() >= 3 && !is.na(sd(rougeL)) && sd(rougeL) > 0,
    W = if(is_testable) shapiro_test(rougeL)$statistic else NA_real_,
    p = if(is_testable) shapiro_test(rougeL)$p.value else NA_real_,
    n = n(),
    .groups="drop"
  ) %>% 
  mutate(metric="ROUGE-L") %>%
  select(-is_testable) # Clean up helper column

# This part will now work correctly
sw_all <- bind_rows(sw_f1, sw_rouge) %>% select(metric, quantization_level, n, W, p)
write_csv(sw_all, file.path(output_dir, "accuracy_normality_shapiro.csv"))
print(sw_all)

# === Attempt normalization before inferential tests ===
eps <- 1e-6
df <- df %>%
  mutate(
    f1_transformed = asin(sqrt(pmin(pmax(f1, 0), 1))),
    rougeL_transformed = asin(sqrt(pmin(pmax(rougeL, 0), 1)))
  )

# Shapiro–Wilk test on transformed data
sw_f1_tf <- df %>%
  filter(task_type=="QA", !is.na(f1_transformed)) %>%
  group_by(quantization_level) %>%
  summarise(
    W = if(n()>=3) shapiro.test(f1_transformed)$statistic else NA_real_,
    p = if(n()>=3) shapiro.test(f1_transformed)$p.value  else NA_real_,
    .groups="drop"
  ) %>% mutate(metric="F1_transformed")

sw_rg_tf <- df %>%
  filter(task_type %in% c("SUMM_SHORT","SUMM_LONG"), !is.na(rougeL_transformed)) %>%
  group_by(quantization_level) %>%
  summarise(
    W = if(n()>=3) shapiro.test(rougeL_transformed)$statistic else NA_real_,
    p = if(n()>=3) shapiro.test(rougeL_transformed)$p.value  else NA_real_,
    .groups="drop"
  ) %>% mutate(metric="ROUGE-L_transformed")

# Combine and save
sw_all_tf <- bind_rows(sw_f1_tf, sw_rg_tf)
write_csv(sw_all_tf, file.path(output_dir, "accuracy_normality_shapiro_transformed.csv"))
print(sw_all_tf)

# Decision: use ANOVA if all p >= 0.05, else Kruskal
normal_f1 <- all(sw_f1_tf$p >= 0.05, na.rm=TRUE)
normal_rg <- all(sw_rg_tf$p >= 0.05, na.rm=TRUE)


# --- Kruskal–Wallis + Dunn post-hoc (QA: F1) ---
qa <- df %>% filter(task_type=="QA", !is.na(f1))
kw_f1 <- if(nrow(qa)>0 && length(unique(qa$quantization_level)) > 1) {
  kruskal.test(f1 ~ quantization_level, data=qa)
} else {
  NULL
}

dunn_f1 <- if(!is.null(kw_f1) && kw_f1$p.value < 0.05) {
  dunnTest(f1 ~ quantization_level, data=qa, method="bonferroni")$res
} else {
  NULL
}

# effect sizes (Cliff's delta) for all pairs
qa_levels <- unique(qa$quantization_level)
pairs_f1 <- if(length(qa_levels) >= 2) t(combn(qa_levels, 2)) else matrix(nrow=0, ncol=2)

es_f1 <- if (nrow(pairs_f1) > 0) {
  apply(pairs_f1, 1, function(p) {
    g1 <- qa$f1[qa$quantization_level==p[1]]
    g2 <- qa$f1[qa$quantization_level==p[2]]
    data.frame(metric="F1", g1=p[1], g2=p[2],
               delta = if(length(g1)>1 && length(g2)>1) cliff.delta(g1, g2)$estimate else NA_real_)
  }) %>% bind_rows()
} else {
  tibble() # Empty tibble if no pairs
}


# --- Kruskal–Wallis + Dunn post-hoc (SUMM: ROUGE-L) ---
sm <- df %>% filter(task_type %in% c("SUMM_SHORT","SUMM_LONG"), !is.na(rougeL))
kw_rg <- if(nrow(sm)>0 && length(unique(sm$quantization_level)) > 1) {
  kruskal.test(rougeL ~ quantization_level, data=sm)
} else {
  NULL
}

dunn_rg <- if(!is.null(kw_rg) && kw_rg$p.value < 0.05) {
  dunnTest(rougeL ~ quantization_level, data=sm, method="bonferroni")$res
} else {
  NULL
}

sm_levels <- unique(sm$quantization_level)
pairs_rg <- if(length(sm_levels) >= 2) t(combn(sm_levels, 2)) else matrix(nrow=0, ncol=2)

es_rg <- if (nrow(pairs_rg) > 0) {
  apply(pairs_rg, 1, function(p) {
    g1 <- sm$rougeL[sm$quantization_level==p[1]]
    g2 <- sm$rougeL[sm$quantization_level==p[2]]
    data.frame(metric="ROUGE-L", g1=p[1], g2=p[2],
               delta = if(length(g1)>1 && length(g2)>1) cliff.delta(g1, g2)$estimate else NA_real_)
  }) %>% bind_rows()
} else {
  tibble() # Empty tibble if no pairs
}

# --- Save stats ---
if(!is.null(kw_f1)) write_csv(
  data.frame(metric="F1", test="Kruskal-Wallis", p_value=kw_f1$p.value),
  file.path(output_dir, "accuracy_kw_overall_f1.csv")
)
if(!is.null(kw_rg)) write_csv(
  data.frame(metric="ROUGE-L", test="Kruskal-Wallis", p_value=kw_rg$p.value),
  file.path(output_dir, "accuracy_kw_overall_rouge.csv")
)
if(!is.null(dunn_f1)) write_csv(dunn_f1, file.path(output_dir, "accuracy_dunn_f1.csv"))
if(!is.null(dunn_rg)) write_csv(dunn_rg, file.path(output_dir, "accuracy_dunn_rouge.csv"))

if(nrow(es_f1) > 0) write_csv(es_f1, file.path(output_dir, "accuracy_effectsizes_f1.csv"))
if(nrow(es_rg) > 0) write_csv(es_rg, file.path(output_dir, "accuracy_effectsizes_rouge.csv"))

# Removed the extra '}' at the end of the file