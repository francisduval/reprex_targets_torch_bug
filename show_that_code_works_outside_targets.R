# An R script to show that everything is working outside targets
library(targets)
library(visNetwork)
library(torch)
library(tidyverse)
library(tarchetypes)
library(luz)

source("R/MyDataset.R")
source("R/NeuralNet.R")

# ===============================================================================================================================

MyDataset(mtcars)
NeuralNet()

# ===============================================================================================================================

mtcars_train_ds <- MyDataset(mtcars)
mtcars_train_dl <- dataloader(mtcars_train_ds, batch_size = 4)

mtcars_valid_ds <- MyDataset(mtcars)
mtcars_valid_dl <- dataloader(mtcars_valid_ds, batch_size = 4)

fit <- 
  NeuralNet %>% 
  setup(
    loss = nnf_mse_loss,
    optimizer = optim_adam,
    metrics = list(luz_metric_mse())
  ) %>%
  luz::fit(
    mtcars_train_dl, 
    epochs = 1, 
    valid_data = mtcars_valid_dl
  )
