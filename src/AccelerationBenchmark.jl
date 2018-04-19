__precompile__() # TODO: Should we precompile?
module AccelerationBenchmark

using OptimTestProblems.MultivariateProblems
using NLSolversBase
using Optim, LineSearches
using DataFrames, DataArrays
using StatPlots, GroupedErrors
using CUTEst
using FileIO

const MVP = MultivariateProblems
const UP = MVP.UnconstrainedProblems

export OptimizationRun, TestSetup, OACCEL2017,
    MinimaApproximator

export createruns, createmeasures,
    createmeasuredataframe, createratiodataframe,
    createviolins, createboxplots, createstatplots,
    createperfprofiles

export RandomizeProblem, RandomizeInitialx, RandomizeInitialxMat,
    randomizeproblem!

const DATADIR = (@__DIR__)*"/../data/"

include("types.jl")
include("api.jl")
include("utils.jl")
include("randomizer.jl")

include("benchmarks/oaccel2017.jl")
include("benchmarks/minimaapproximator.jl")

end # module
