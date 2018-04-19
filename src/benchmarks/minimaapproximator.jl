"Tooling for recording minima for test problems"
module MinimaApproximator

using CSV, DataFrames

"Return TestSetup used to approximate minima for test problems."
function defaultminimumsearchts()
    solvers = [
        LBFGS(),
        Newton(),
        OACCEL()
    ]
    solvernames = [
        "L-BFGS",
        "Newton",
        "O-ACCEL"
    ]
    stoptype = :GradientTolerance
    stoptol  = 0.0
    timelog  = 0 # timelog = 1 can cause very inaccurate timings due to compilation and garbage collection
    maxiter  = 2000
    timelimit = NaN
    TestSetup(solvers,solvernames,stoptype,stoptol,timelog,maxiter,timelimit)
end

"Create table recording the minima from performance measure data frame"
function createminimatable(mdf::DataFrame)
    mindf = DataFrame(Problem = mdf[:Problem],
                      Minimum = mdf[:fval],
                      GradientInfNorm = mdf[:gnorm])
    mindf = purgeminimatable(mindf::DataFrame)

    return mindf
end

"Return only the lowest value recorded each problem in the minimatable."
function purgeminimatable(mindf::DataFrame)
    colidx = copy(mindf.colindex)
    delete!(colidx, :Problem)

    retval = by(mindf, :Problem) do df
        df[findfirst(x -> x == minimum(df[:Minimum]), df[:Minimum]), colidx.names]
    end
    retval
end

"Purge duplicate entries for a Problem in csvstore"
function purgeminimatable(csvstore::AbstractString = Pkg.dir("AccelerationBenchmark")*"/data/cutestmins.csv")
    df = purgeminimatable(CSV.read(csvstore))
    CSV.write(csvstore, df; append=false, header=true)
end

"""
Helper method for approximating the minima of the CUTEst models in `cutestnames`,
and appending it to `csvstore`.

Return minima DataFrame, performance measure DataFrame, and vector of OptimizationRuns.
"""
function approximatecutestminima(cutestnames::AbstractVector{<:AbstractString},
                                 ts::TestSetup = defaultminimumsearchts(),
                                 csvstore::AbstractString = Pkg.dir("AccelerationBenchmark")*"/data/cutestmins.csv";
                                 saveindividualoruns::Bool = true)
    oruns = createmeasures(cutestnames, ts; saveindividualoruns = saveindividualoruns)
    mdf = createmeasuredataframe(oruns)
    mindf = createminimatable(mdf)

    if !isempty(csvstore)
        CSV.write(csvstore, mdf; append=true, header=true)
        # If the user wants, they can later remove duplicates with `purgeminimatable`.
    end

    return mindf, mdf, oruns
end

end
