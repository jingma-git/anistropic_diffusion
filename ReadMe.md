# references
1. Scale-Space and Edge Detection Using Anisotropic Diffusion
2. Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow

# Result
1. Double precision is 2 times slower than single precision
2. Explicit method: lambda <= 0.1, one_step_time=0.001s
3. Implicit method: the test max_value is 100
4. 1 step implicit method(lambda=100, 1.7s) == 1000 explicit method(lambda=0.1, 1~2s)
5. Most of time of implicit method spent on "coeff build(0.06s)" and "factorize(1.64s)".

# Insight
1. The key to get a plausible result is how to design "diffusion coefficient" because it guides how the gradient/flow disperse.

# ToDo
1. Make the program parrallel