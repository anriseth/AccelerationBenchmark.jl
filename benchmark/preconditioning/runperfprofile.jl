using AccelerationBenchmark
using JLD, DataFrames, StatPlots
pyplot()

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

"Performance profiles from the Moré tests"
function more_tests(;xmax::Number = NaN, solvers = Vector{String}(0))
    fnames = ["data/"*fname for fname in searchdir("data", "jld")]
    fnames = fnames[.!contains.(fnames, "cutest")]
    mdf = reduce(append!, [load(fname, "mdf") for fname in fnames])
    rdf = AccelerationBenchmark.createratiodataframe(mdf, solvers)

    # TODO: GroupedErrors is using a uniform spread in the x-values (\tau-values),
    # so there is quite low resolution for small x-values.
    # I need to deal with this somehow, and combine that with a downsampling approach.
    # Alternatively, we can create fevals_AG.csv and run it through "mergealldata.jl" with BenchmarkProfiles.

    syms = [:fcalls, :gcalls, :Iterations, :CPUtime]
    for sym in syms
        rdf[sym] .=  NaNMath.log2.(rdf[sym])
    end
    maxfun(x) = maximum(x)

    pltf = AccelerationBenchmark.createperfprofiles(rdf, :fcalls; xlabel="f-calls, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)
    plti = AccelerationBenchmark.createperfprofiles(rdf, :Iterations; xlabel="Iterations, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)
    pltg = AccelerationBenchmark.createperfprofiles(rdf, :gcalls; xlabel="g-calls, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)
    pltc = AccelerationBenchmark.createperfprofiles(rdf, :CPUtime; xlabel="CPU time, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)

    if isfinite(xmax)
        xlims!(pltf,0,xmax);xlims!(pltg,0,xmax);xlims!(pltc,0,xmax);xlims!(plti,0,xmax)
    end
    plt = plot(pltf,pltg,pltc,plti)
end

function cutest_tests(fname::AbstractString = "data/cutest_20_10000.jld",
                      xmax::Number = NaN;
                      solvers = Vector{String}(0))
    mdf = load(fname, "mdf")

    rdf = AccelerationBenchmark.createratiodataframe(mdf, solvers)

    # TODO: GroupedErrors is using a uniform spread in the x-values (\tau-values),
    # so there is quite low resolution for small x-values.
    # I need to deal with this somehow, and combine that with a downsampling approach.
    syms = [:fcalls, :gcalls, :Iterations, :CPUtime]
    for sym in syms
        rdf[sym] .=  NaNMath.log2.(rdf[sym])
    end
    maxfun(x) = maximum(x)

    pltf = AccelerationBenchmark.createperfprofiles(rdf, :fcalls; xlabel="f-calls, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)
    plti = AccelerationBenchmark.createperfprofiles(rdf, :Iterations; xlabel="Iterations, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)
    pltg = AccelerationBenchmark.createperfprofiles(rdf, :gcalls; xlabel="g-calls, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)
    pltc = AccelerationBenchmark.createperfprofiles(rdf, :CPUtime; xlabel="CPU time, log₂-scale",
                                                    xscale=:identity, maxfun = maxfun)

    if isfinite(xmax)
        xlims!(pltf,0,xmax);xlims!(pltg,0,xmax);xlims!(pltc,0,xmax);xlims!(plti,0,xmax)
    end
    plt = plot(pltf,pltg,pltc,plti)
end

# solvers=Vector{String}(0)
#solvers=["O-ACCEL-10", "O-ACCEL-20"]
#solvers=["O-ACCEL-10", "GD"]
plt = more_tests(;solvers=solvers)
savefig(plt, "perfprof_more_tests.svg")
savefig(plt, "perfprof_more_tests.png")


minvar = 20; maxvar = 100000
pltc = cutest_tests("data/cutest_$(minvar)_$(maxvar).jld", NaN;
                    solvers = solvers)
savefig(pltc, "perfprof_cutest_$(minvar)_$(maxvar).svg")
savefig(pltc, "perfprof_cutest_$(minvar)_$(maxvar).png")
