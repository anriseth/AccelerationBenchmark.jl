# Performance profiles to compare L-BFGS with and without scaling
Code to compare L-BFGS for a variety of line searches, both with and
without the inverse Hessian scaling from Nocedal & Wright (2nd ed), Equation (7.20).

Run `runtests.jl` to generate the data.
Run the Moré et al. tests by enabling `run_tests_more`, and
run CUTEst tests by enabling `run_tests_cutest`.
It supports parallel execution with Julia. For example, run with eight
workers, use `julia -p 8 runtests.jl`.


Run `runperfprofile.jl` to generate performance profile plots.
