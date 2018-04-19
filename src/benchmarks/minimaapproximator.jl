"Tooling for recording minima for test problems"
module MinimaApproximator
using CSV, DataFrames
using Optim, LineSearches
using AccelerationBenchmark, OptimTestProblems.MultivariateProblems, CUTEst
import NaNMath

const MINIMACSV = AccelerationBenchmark.DATADIR*"/cutestmins.csv"


"Return TestSetup used to approximate minima for test problems."
function defaultminimumsearchts()
    ls = BackTracking(order=2)
    solvers = [
        LBFGS(linesearch = ls),
        Newton(linesearch = ls),
        OACCEL(linesearch = ls)
    ]
    solvernames = [
        "L-BFGS",
        "Newton",
        "O-ACCEL"
    ]
    stoptype = :GradientTol
    stoptol  = 0.0
    timelog  = 0 # timelog = 1 can cause very inaccurate timings due to compilation and garbage collection
    maxiter  = 2000
    timelimit = 200.0 # In seconds. Mainly prevents Newton from going on forever on bigger problems
    TestSetup(solvers,solvernames,stoptype,stoptol,timelog,maxiter,timelimit)
end

"Create table recording the minima from performance measure data frame"
function createminimatable(mdf::DataFrame)
    mindf = DataFrame(Problem = mdf[:Problem],
                      Minimum = mdf[:fval],
                      GradientInfNorm = mdf[:gnorm])
    mindf = purgeminimatable(mindf)

    return mindf
end

"Return only the lowest value recorded each problem in the minimatable."
function purgeminimatable(mindf::DataFrame)
    colidx = copy(mindf.colindex)
    delete!(colidx, :Problem)

    retval = by(mindf, :Problem) do df
        df[findfirst(x -> x == NaNMath.minimum(df[:Minimum]), df[:Minimum]), colidx.names]
    end
    retval
end

"Purge duplicate entries for a Problem in csvstore"
function purgeminimatable(csvstore::AbstractString = MINIMACSV)
    if isfile(csvstore)
        df = purgeminimatable(CSV.read(csvstore))
        CSV.write(csvstore, df; append=false, header=true)
    else
        warn("The file `csvstore` not exist.\n `csvstore = $csvstore`.")
    end
end

"""
Helper method for approximating the minima of the CUTEst models in `cutestnames`,
and appending it to `csvstore`.

Return minima DataFrame, performance measure DataFrame, and vector of OptimizationRuns.
"""
function approximatecutestminima(cutestnames::AbstractVector{<:AbstractString},
                                 ts::TestSetup = defaultminimumsearchts(),
                                 csvstore::AbstractString = MINIMACSV;
                                 saveindividualoruns::Bool = true)
    oruns = createmeasures(cutestnames, ts; saveindividualoruns = saveindividualoruns)
    mdf = createmeasuredataframe(oruns)
    mindf = createminimatable(mdf)

    if !isempty(csvstore)
        append = isfile(csvstore)
        CSV.write(csvstore, mindf; append=append, header=true)
        # If the user wants, they can later remove duplicates with `purgeminimatable`.
    end

    return mindf, mdf, oruns
end

" Fetch stored approximate minimum. "
function solver_optimum(name::AbstractString,
                        mincsv::AbstractString = MINIMACSV)
    retval = NaN
    try
        mindf = isfile(mincsv) ? CSV.read(mincsv) : DataFrame()

        if haskey(mindf.colindex, :Problem)
            idx = findfirst(x -> x == name, mindf[:Problem])
            if idx > 0
                retval = mindf[idx, :Minimum]
            end
        end
    catch e
        AccelerationBenchmark.soft_error(e)
        retval = NaN
    end
    return retval
end

" Fetch stored approximate minimum. "
solver_optimum(p::OptimizationProblem,
               mincsv::AbstractString = MINIMACSV) =
                   solver_optimum(p.name, mincsv)

" Fetch stored approximate minimum. "
solver_optimum(nlp::CUTEstModel,
               mincsv::AbstractString = MINIMACSV) =
                   solver_optimum(AccelerationBenchmark.CUTEstOPname(nlp), mincsv)




end
