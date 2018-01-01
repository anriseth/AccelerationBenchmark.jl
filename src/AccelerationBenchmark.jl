#=
TODO:
-
=#

module AccelerationBenchmark

using OptimTestProblems.UnconstrainedProblems
using NLSolversBase
using Optim, LineSearches
using DataFrames, DataArrays
using StatPlots, GroupedErrors
using CUTEst

export OptimizationRun, TestSetup, TestBench

export createruns, createruns!, createmeasures,
    createmeasuredataframe, createratiodataframe,
    createviolins, createboxplots, createstatplots,
    createperfprofiles


include("types.jl")
include("api.jl")
include("utils.jl")
include("randomizer.jl")

include("benchmarks/testbench.jl")

end # module
