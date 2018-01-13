# Performance profiles to evaluate the introduction of preconditioners to Optim
Code to evaluate performance of NGMRES and OACCEL preconditioned with existing
algorithms in Optim.
We use Gradient Descent preconditioning, and see how it improves performance.
For comparison, L-BFGS results are also calculated.

Run `runtests.jl` to generate the data.
Run the Moré et al. tests by enabling `create_tests_more`, and
run CUTEst tests by enabling `create_tests_cutest`.
It supports parallel execution with Julia. For example, run with eight
workers, use `julia -p 8 runtests.jl`.

Run `runperfprofile.jl` to generate performance profile plots.
