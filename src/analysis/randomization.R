datasets <- read.csv('../datasets.csv')

# Check that we have 30 subjects
N <- 30
if (nrow(datasets) != N) {
  print(datasets)
}

# Perform randomization

# for R_i = 0, we assign 3 to treatment
m_block_0 <- 3
# for R_i = 1, we assign 11 to treatment
m_block_1 <- 11

set.seed(0527)

block_0_treated <- sample(which(datasets$richness_block == 0), m_block_0)
block_1_treated <- sample(which(datasets$richness_block == 1), m_block_1)

Z <- rep(0, N)
Z[block_0_treated] <- 1
Z[block_1_treated] <- 1

assignments <- data.frame(dataset=datasets$Dataset, Z=Z)
write.csv(assignments, '../assignments.csv')