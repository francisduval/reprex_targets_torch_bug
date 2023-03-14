NeuralNet = nn_module(
  "NeuralNet",
  
  initialize = function() {
    self$linear1 = nn_linear(5, 2)
    self$linear2 = nn_linear(2, 1)
  },
  
  forward = function(x) {
    x %>%
      self$linear1() %>%
      nnf_relu() %>%
      self$linear2()
  }
)
