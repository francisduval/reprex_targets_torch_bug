MyDataset <- dataset(
  name = "MyDataset",
  
  initialize = function(df) {
    self$x <- df %>% dplyr::select(hp, cyl, disp, gear, carb) %>% as.matrix() %>% torch_tensor()
    self$y <- torch_tensor(df$mpg) %>% torch_unsqueeze(dim = 2)
    
  },
  
  .getitem = function(i) {
    list(x = self$x[i, ], y = self$y[i])
  },
  
  .length = function() {
    self$y$size()[[1]]
  }
)
