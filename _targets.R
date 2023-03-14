library(targets)
library(visNetwork)
library(torch)
library(tidyverse)
library(tarchetypes)
library(luz)

source("R/MyDataset.R")
source("R/NeuralNet.R")

list(
  # Dataset object
  tar_target(torch_target_1, MyDataset(mtcars)), # no error, but cannot read with tar_read(torch_target_1)
  tar_target(torch_target_2, MyDataset(mtcars), format = "torch"), # errors out
  tar_torch(torch_target_3, MyDataset(mtcars)), # errors out
  
  # ----------
  
  # nn_module object
  tar_target(torch_target_4, NeuralNet()), # no error, but cannot read with tar_read(torch_target_4)
  tar_target(torch_target_5, NeuralNet(), format = "torch"), # this works out
  tar_torch(torch_target_6, NeuralNet()), # this works out
  
  # ----------
  
  # luz_module_fitted
  tar_target(
    fitted_luz_1, # no error, but cannot read with tar_read(fitted_luz_1)
    {
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

      return(fit)
    }
  ),
  
  tar_target(
    fitted_luz_2, # errors out
    {
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

      return(fit)
    },
    format = "torch"
  ),
  
  tar_torch(
    fitted_luz_3, # errors out
    {
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
      
      return(fit)
    }
  )
)

# Obviously, all of this work when executed outside targets. See show_that_code_works_outside_targets.R.
