# Performance profiles for O-ACCEL
Code to reproduce the performance profiles and data for the tables in
the paper
"A.N. Riseth, **Objective acceleration for unconstrained optimization**, 2018"

Run `runtests.jl` to generate the data. It supports parallel execution
with Julia. For example, run with eight workers, use `julia -p 8 runtests.jl`.

Run `runperfprofile.jl` to generate performance profile plots.
