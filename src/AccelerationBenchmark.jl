#=
TODO:
-
=#

module AccelerationBenchmark

using OptimTestProblems.UnconstrainedProblems
using NLSolversBase
using Optim, LineSearches
using DataFrames

include("types.jl")
include("api.jl")
include("utils.jl")

end # module
