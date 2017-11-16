#=
TODO:
-
=#

module AccelerationBenchmark

using OptimTestProblems.UnconstrainedProblems
using NLSolversBase
using Optim, LineSearches
using DataFrames, DataArrays
using StatPlots

export OptimizationRun, TestBench

export createruns, createruns!, createmeasuredataframe,
    createratiodataframe, createviolins, createboxplots

include("types.jl")
include("api.jl")
include("utils.jl")

include("benchmarks/testbench.jl")

end # module
