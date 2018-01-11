using AccelerationBenchmark
using JLD, DataFrames, StatPlots, NaNMath, LaTeXStrings

searchdir(path,key) = filter(x->contains(x,key), readdir(path))
mdf = reduce(append!, [load(fname, "mdf") for fname in ["data/"*fname for fname in searchdir("data", "jld")]])
rdf = AccelerationBenchmark.createratiodataframe(mdf)


# TODO: GroupedErrors is using a uniform spread in the x-values (\tau-values),
# so there is quite low resolution for small x-values.
# I need to deal with this somehow, and combine that with a downsampling approach.
# Alternatively, we can create fevals_AG.csv and run it through "mergealldata.jl" with BenchmarkProfiles.

rdf[:fcalls] .=  NaNMath.log2.(rdf[:fcalls])
plt = AccelerationBenchmark.createperfprofiles(rdf,:fcalls; xscale=:linear,
                                              xlabel=L"$\tau$ ($\log_2$ scale)")
