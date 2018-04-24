using AccelerationBenchmark
using FileIO, DataFrames, StatPlots, NaNMath, LaTeXStrings

#pyplot()
pgfplots()

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

"Performance profiles from the Moré tests"
function more_tests()
    mdf = reduce(append!, [load(fname, "mdf") for fname in ["data/"*fname for fname in searchdir("data", "jld")]])
    rdf = AccelerationBenchmark.createratiodataframe(mdf)


    # TODO: GroupedErrors is using a uniform spread in the x-values (\tau-values),
    # so there is quite low resolution for small x-values.
    # I need to deal with this somehow, and combine that with a downsampling approach.
    # Alternatively, we can create fevals_AG.csv and run it through "mergealldata.jl" with BenchmarkProfiles.

    maxfun(x) = 1.1maximum(x)

    rdf[:fcalls] .=  NaNMath.log2.(rdf[:fcalls])
    plt = AccelerationBenchmark.createperfprofiles(rdf,:fcalls; xscale=:linear,
                                                   xlabel=L"$\tau$ ($\log_2$ scale)")

    return plt
end

function cutest_tests(fname::AbstractString = "data/cutest_our_selected.jld2",
                      xmax::Number = NaN)
    mdf = load(fname, "mdf")

    rdf = AccelerationBenchmark.createratiodataframe(mdf)

    # TODO: GroupedErrors is using a uniform spread in the x-values (\tau-values),
    # so there is quite low resolution for small x-values.
    # I need to deal with this somehow, and combine that with a downsampling approach.
    syms = [:fcalls, :gcalls, :Iterations, :CPUtime]
    for sym in syms
        rdf[sym] .=  NaNMath.log2.(rdf[sym])
    end
    maxfun(x) = 1.1maximum(x)

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

#plt_more = more_tests()

plt_cutest = cutest_tests("data/cutest_our_selected.jld2", NaN)
