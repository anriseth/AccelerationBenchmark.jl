__precompile__()
module AccelerationBenchmark

using OptimTestProblems.UnconstrainedProblems
using NLSolversBase
using Optim, LineSearches
using DataFrames, DataArrays
using StatPlots, GroupedErrors
using CUTEst

export OptimizationRun, TestSetup, OACCEL2017

export createruns, createmeasures,
    createmeasuredataframe, createratiodataframe,
    createviolins, createboxplots, createstatplots,
    createperfprofiles

export RandomizeProblem, RandomizeInitialx, RandomizeInitialxMat,
    randomizeproblem!


include("types.jl")
include("api.jl")
include("utils.jl")
include("randomizer.jl")

include("benchmarks/oaccel2017.jl")

end # module
