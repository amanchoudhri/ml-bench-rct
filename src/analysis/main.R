# Required packages
library(tidyverse)  # for data wrangling
library(sandwich)   # for robust standard errors
library(kableExtra) # for nice tables
library(ggplot2)    # for plotting
library(texreg)     # for LaTeX regression tables
library(ri2)        # for randomization inference

# Read data
datasets <- read.csv("../datasets.csv")
assignments <- read.csv("../assignments.csv")

# Merge datasets
data <- datasets %>%
  inner_join(assignments, by = c("Dataset"="dataset")) %>%
  mutate(
    block = richness_block,
    # Log transform image size
    log_img_size = log(img_size),
  )

# Calculate assignment probabilities by block
probs <- data %>%
  group_by(block) %>%
  summarize(
    n = n(),
    n_treated = sum(Z),
    p = n_treated / n
  )

# Join probabilities back and calculate weights
data <- data %>%
  left_join(probs, by = "block") %>%
  mutate(weight = Z/p + (1-Z)/(1-p))

# Function to create regression tables
create_reg_table <- function(results, filename) {
  texreg(
    results,
    file = paste0("doc/generated/", filename, ".tex"),
    booktabs = TRUE,
    caption = "Treatment Effect Estimates",
    label = paste0("tab:", filename),
    include.ci = FALSE,
    custom.model.names = c("Block-Adjusted", "Covariate-Adjusted"),
    custom.coef.names = c(
      "(Intercept)" = "Intercept",
      "Z" = "Treatment (ViT)",
      "log_img_size" = "Log Image Size"
    )
  )
}

# After we observe outcomes Y, we'll run:

# Function to analyze outcomes
analyze_outcomes <- function(data) {
  # Fit models
  m1 <- lm(Y ~ Z, data = data, weights = weight)
  m2 <- lm(Y ~ Z + log_img_size, data = data, weights = weight)
  
  # Create results table
  create_reg_table(list(m1, m2), "treatment_effects")
  
  # Randomization inference
  ri_out <- conduct_ri(
    Y ~ Z + log_img_size, 
    blocks = ~block,
    declaration = declare_ra(blocks = data$block, 
                             block_prob = probs$p)
  )
  
  # Create diagnostic plots
  p1 <- ggplot(data, aes(x = log_img_size, y = Y, color = factor(Z))) +
    geom_point() +
    geom_smooth(method = "lm") +
    labs(x = "Log Image Size", y = "Accuracy", color = "Treatment") +
    theme_minimal()
  
  ggsave("doc/generated/outcome_by_size.pdf", p1)
  
  # Balance plot
  p2 <- ggplot(data, aes(x = log_img_size, fill = factor(Z))) +
    geom_density(alpha = 0.5) +
    labs(x = "Log Image Size", y = "Density", fill = "Treatment") +
    theme_minimal()
  
  ggsave("doc/generated/covariate_balance.pdf", p2)
  
  return(list(
    models = list(m1, m2),
    ri = ri_out,
    plots = list(p1, p2)
  ))
}