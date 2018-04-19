function getopts(gt::GradientTolerance{T}, maxiter::Int, time_limit::Real = NaN;
                 kwargs...) where T
    # TODO: Figure out a way to make passing options more flexible (time_limit etc.)
    opts = Optim.Options(f_tol = zero(T), x_tol = zero(T),
                         g_tol = gt.g0norm*gt.tol,
                         allow_f_increases = true,
                         iterations = maxiter,
                         time_limit = time_limit,
                         kwargs...)#,
                         #show_trace=true,extended_trace=true, store_trace=true)
    return opts
end

function getopts(ft::FunctionTolerance{T}, maxiter::Int, time_limit::Real = NaN;
                 kwargs...) where T
    # TODO: Figure out a way to make passing options more flexible (time_limit etc.)
    cb = FunctionStop(ft.fL + ft.tol*(ft.f0 - ft.fL))
    opts = Optim.Options(f_tol = zero(T), x_tol = zero(T), g_tol = zero(T),
                         allow_f_increases = true,
                         iterations = maxiter, callback = cb,
                         time_limit = time_limit;
                         kwargs...)
    return opts
end

"Return time elapsed and result of expression"
macro my_elapsed(ex)
    quote
        local t0 = time_ns()
        local val = $(esc(ex))
        (time_ns()-t0)/1e9, val
    end
end


function runproblem(df, x0, solver::Optim.AbstractOptimizer, solvername::AbstractString,
                    problemname::AbstractString,
                    metrictol::StopTolerance;
                    numrecordtime::Int = 0, maxiter::Int = 1000,
                    time_limit::Real = NaN)
    opts = getopts(metrictol, maxiter, time_limit)

    f0, g0 = value_gradient!(df, x0)
    g0norm = norm(g0, Inf)
    try
        # TODO: Does this help for precompilation at all?
        # This can be unnecessarily costly if the line search goes bananas
        optimize(df, x0, solver, Optim.Options(iterations=1))
    catch e
        warn(e)
    end

    reset!(df) # Get correct f_calls numbering
    sname = isempty(solvername) ? summary(solver) : solvername
    orun = OptimizationRun(0, 0, 0, 0, 0.0, f0, f0, g0norm, g0norm,
                           sname, problemname, metrictol)
    try
        runtime, r = @my_elapsed optimize(df, x0, solver, opts)
        @show Optim.f_calls(r), Optim.g_calls(r)
        for k = 2:numrecordtime
            # TODO: Do this with BenchmarkTools?
            reset!(df) # Reset df to get f_calls etc. correct
            tmptime, tmpr = @my_elapsed optimize(df, x0, solver, opts)
            if tmptime < runtime
                runtime = tmptime
                r = tmpr # In case previous r hit time_limit
            end
        end

        orun = OptimizationRun(Optim.iterations(r), Optim.f_calls(r), Optim.g_calls(r),
                               Optim.h_calls(r), runtime, Optim.minimum(r), f0,
                               Optim.g_residual(r), g0norm, sname, problemname,
                               metrictol)
    catch e
        warn(e)
    end
    return orun
end


"""
Save individual data, default assumption is that the data is an
`OptimizationRun`, or an array of `OptimizationRun`s
"""
function savedata(data,
                  name::AbstractString,
                  basedir::AbstractString = DATADIR*"/oruns/")
    outname = basedir*"/"*name*".jld2"
    try
        save(outname, Dict(name => data))
    catch e
        warn(e)
    end
end

"""
Loop over solvers for OptimizationProblem `prob` and optimize until a given `stoptolerance`.

Return vector of `OptimizationRun`s for a given problem.
Optionally include runtime (runs optimization multiple times per solver).
"""
function createruns(prob::OptimizationProblem, problemname::AbstractString,
                    solvers::AbstractVector, solvernames::AbstractVector{<:AbstractString},
                    tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Int = 0,
                    maxiter::Int = 1000; time_limit::Real = NaN,
                    verbose::Bool = true,
                    saveindividualoruns::Bool = false) where T <: Real
    # TODO: Is it problematic for specialized compilation to use stoptype::Symbol?
    verbose && println("Running $problemname ...")
    local x0, f0, g0norm # Ensure variables stay in method scope
    try
        x0 = initial_x(prob)
        df0 = optim_problem(prob)
        f0, g0 = value_gradient!(df0, x0)
        g0norm = norm(g0, Inf)
    catch e
        warn(e)
        oruns = Vector{OptimizationRun{typeof(time_ns()/1e9),T}}(0)
        if saveindividualoruns
            savedata(oruns, problemname)
        end
        return oruns
    end

    if stoptype == :GradientTolRelative
        metrictol = GradientTolerance(tol,g0norm)
    elseif stoptype == :GradientTol
        metrictol = GradientTolerance(tol,one(T))
    elseif stoptype == :FunctionTolRelative
        fL = solver_optimum(prob)
        @assert isfinite(fL) # TODO: create function to find solver_optimum
        metrictol = FunctionTolerance(tol, f0, fL)
    elseif stoptype == :FunctionTol
        fL = solver_optimum(prob)
        @assert isfinite(fL)
        metrictol = FunctionTolerance(tol, one(fL) + fL, fL)
    else
        Base.error("The parameter `stoptype` must be one of :GradientTolRelative, :GradientTol, :FunctionTolRelative, or :FunctionTol.")
    end

    oruns = Vector{OptimizationRun{typeof(time_ns()/1e9),T}}(length(solvers))
    for (k, solver) in enumerate(solvers)
        df = optim_problem(prob)
        try
            oruns[k] = runproblem(df, x0, solver, solvernames[k],
                                  problemname, metrictol;
                                  numrecordtime=timelog, maxiter = maxiter,
                                  time_limit = time_limit)
        catch probsolve
            warn(probsolve)
            oruns[k] = OptimizationRun(0,0,0,0,zero(time_ns()/1e9),
                                       f0,f0,g0norm,g0norm,
                                       solvernames[k],problemname,false)
        end
    end
    verbose && println("Finished $problemname.")

    if saveindividualoruns
        savedata(oruns, problemname)
    end
    return oruns
end

"""
Loop over solvers for CUTEst problem `cutestname` and optimize until a given `stoptolerance`.

Return vector of `OptimizationRun`s for a given problem.
Optionally include runtime (runs optimization multiple times per solver).
"""
function createruns(cutestname::AbstractString,
                    solvers::AbstractVector, solvernames::AbstractVector{<:AbstractString},
                    tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Int = 0,
                    maxiter::Int = 1000; time_limit::Real = NaN,
                    decode::Bool = false,
                    verbose::Bool = true,
                    saveindividualoruns::Bool = false
                    ) where T <: Real
    verbose && println("Loading $cutestname ...")
    local oruns
    try
        nlp = CUTEstModel(cutestname, decode=decode)
        prob = optimizationproblem(nlp)
        oruns = createruns(prob, prob.name, solvers, solvernames,
                           tol, stoptype, timelog, maxiter;
                           verbose = verbose, time_limit = time_limit
                           saveindividualoruns = saveindividualoruns)
        finalize(nlp)
    catch e
        warn(e)
        oruns = Vector{OptimizationRun{typeof(time_ns()/1e9),T}}(0)
    end
    return oruns
end
createruns(cutestname::AbstractString, ts::TestSetup;
           saveindividualoruns::Bool = false) =
    createruns(cutestname, ts.solvers, ts.solvernames,
               ts.stoptol, ts.stoptype, ts.timelog, ts.maxiter;
               time_limit = ts.timelimit,
               saveindividualoruns = saveindividualoruns)


"""
Create performance measures of a list of solvers on a collection of OptimizationProblems

Returns a vector of `OptimizationRun`s.
"""
function createmeasures(problems::AbstractVector{<:OptimizationProblem}, problemnames::AbstractVector{<:AbstractString},
                        solvers::AbstractVector, solvernames::AbstractVector{<:AbstractString},
                        tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Int = 0,
                        maxiter::Int = 1000; time_limit::Real = NaN,
                        saveindividualoruns::Bool = false) where T <: Real

    oruns = reduce(append!, pmap(k->createruns(problems[k], problemnames[k],
                                               solvers, solvernames, tol,
                                               stoptype, timelog, maxiter;
                                               time_limit = time_limit,
                                               saveindividualoruns = saveindividualoruns),
                                 1:length(problems)))
    return oruns
end

createmeasures(problems::AbstractVector{<:OptimizationProblem}, problemnames::AbstractVector{<:AbstractString},
               ts::TestSetup;
               saveindividualoruns::Bool = false) =
                   createmeasures(problems, problemnames, ts.solvers, ts.solvernames,
                                  ts.stoptol, ts.stoptype, ts.timelog, ts.maxiter;
                                  time_limit = ts.timelimit,
                                  saveindividualoruns = saveindividualoruns)

"""
Create performance measures of a list of solvers on a collection of CUTEst problems.

Currently, the CUTEst models are run with the default parameters.

Returns a vector of `OptimizationRun`s.
"""
function createmeasures(cutestnames::AbstractVector{<:AbstractString},
                        solvers::AbstractVector, solvernames::AbstractVector{<:AbstractString},
                        tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Int = 0,
                        maxiter::Int = 1000; time_limit::Real = NaN,
                        saveindividualoruns::Bool = true) where T <: Real
    oruns = reduce(append!, pmap(k->createruns(cutestnames[k],
                                               solvers, solvernames, tol,
                                               stoptype, timelog, maxiter;
                                               decode = false,
                                               time_limit = time_limit,
                                               saveindividualoruns = saveindividualoruns),
                                 1:length(cutestnames)))
    return oruns
end

createmeasures(cutestnames::AbstractVector{<:AbstractString},
               ts::TestSetup;
               saveindividualoruns::Bool = false) =
                   createmeasures(cutestnames, ts.solvers, ts.solvernames,
                                  ts.stoptol, ts.stoptype, ts.timelog, ts.maxiter;
                                  time_limit = ts.timelimit,
                                  saveindividualoruns = saveindividualoruns)


"""
Take a vector of `OptimizationRun`s representing performance measures and
convert it to a DataFrame of performance measures
"""
function createmeasuredataframe(oruns::Vector{OptimizationRun{T,Tf}}) where T where Tf
    n         = length(oruns)
    snames    = Vector{String}(n)
    pnames    = similar(snames)
    iters     = Vector{Int}(n)
    fcallarr  = Vector{Int}(n)
    gcallarr  = similar(fcallarr)
    hcallarr  = similar(fcallarr)
    cputimes  = zeros(T, n)
    fvalarr   = zeros(Tf, n)
    f0arr     = similar(fvalarr)
    gnormarr  = similar(fvalarr)
    g0normarr = similar(fvalarr)
    succarr   = Vector{Bool}(n)

    for (k, orun) in enumerate(oruns)
        pnames[k]    = orun.problemname
        snames[k]    = orun.solvername
        iters[k]     = orun.iterations
        fcallarr[k]  = orun.fevals
        gcallarr[k]  = orun.gevals
        hcallarr[k]  = orun.hevals
        cputimes[k]  = orun.cputime
        fvalarr[k]   = orun.fval
        f0arr[k]     = orun.f0
        gnormarr[k]  = orun.gnorm
        g0normarr[k] = orun.g0norm
        succarr[k]   = orun.success
    end

    dframe = DataFrame(Problem = pnames, Solver = snames, Iterations = iters,
                       fcalls = fcallarr, gcalls = gcallarr, hcalls = hcallarr,
                       CPUtime = cputimes, fval = fvalarr, f0 = f0arr,
                       gnorm = gnormarr, g0norm = g0normarr, Success = succarr)
end

" Return performance ratios x / minimum(x), but set elements where `z[k] = false` to `Inf`"
function _nanratio(x, z)
    retval = x ./ minimum(x) # ./ deals with some issues
    # Bool needed since introducing Missings.Missing data type
    retval[Bool.(.!z .| isnan.(retval) .| ismissing.(retval))] = Inf # Should we use NA here?
    return retval
end

"Create a DataFrame of performance ratios from a dataframe of performance measures."
function createratiodataframe(dframe::DataFrame, solvers::AbstractVector{<:AbstractString} = Vector{String}(0))
    if isempty(solvers)
        idx = Colon()
    else
        idx = broadcast(x -> x ∈ solvers, dframe[:Solver])
    end

    by(dframe[idx, :], :Problem) do dfr
        DataFrame(Solver = dfr[:Solver],
                  fcalls = _nanratio(dfr[:fcalls], dfr[:Success]),
                  gcalls = _nanratio(dfr[:gcalls], dfr[:Success]),
                  hcalls = _nanratio(dfr[:hcalls], dfr[:Success]),
                  CPUtime = _nanratio(dfr[:CPUtime], dfr[:Success]),
                  Iterations = _nanratio(dfr[:Iterations], dfr[:Success]),
                  Success = dfr[:Success])
    end
end

"Create a data frame with performance ratios from a vector of `OptimizationRun`s"
createratiodataframe(oruns::Vector{OptimizationRun},
                     solvers::AbstractVector{<:AbstractString} = Vector{String}(0)) =
                         createratiodataframe(createmeasuredataframe(oruns), solvers)


"""
Ensure all performance metrics/ratios are finite, by setting
non-finite value and failures to a "failure" value according to
the rule `maxfun` (default x->2*maximum(x), taken over finite values).
"""
function makefinite(rdf::DataFrame, maxfun = x-> 2*maximum(x))
    cprdf = deepcopy(rdf)
    makefinite!(cprdf, maxfun)
    return cprdf
end

"""
Ensure all performance metrics/ratios are finite, by setting
non-finite value and failures to a "failure" value according to
the rule `maxfun` (default x->2*maximum(x), taken over finite values).
"""
function makefinite!(rdf::DataFrame, maxfun = x-> 2*maximum(x))
    pltvals = [:fcalls, :gcalls,:CPUtime,:Iterations]
    for k in 1:length(pltvals)
        finidx = convert(Array{Bool}, isfinite.(rdf[pltvals[k]]))
        if any(finidx)
            rdf[.!finidx, pltvals[k]] = maxfun(rdf[finidx,pltvals[k]])
        else
            rdf[.!finidx, pltvals[k]] = -1.0
        end
    end
end

"Create violin plot of performance metric/ratios grouped by solver."
createviolins(rdf::DataFrame, yscale::Symbol = :log2) = createstatplots(rdf,violin; yscale = yscale, kwargs...)

"Create box plot of performance metric/ratios grouped by solver."
createboxplots(rdf::DataFrame, yscale::Symbol = :log2) = createstatplots(rdf,boxplot; yscale = yscale, kwargs...)

"Create plot of performance metric/ratios grouped by solver. Defaults to violin plots."
function createstatplots(rdf::DataFrame, fun::Function = violin;
                         yscale = :log2, maxfun = x -> 2*maximum(x),
                         kwargs...)
    pltlabels = ["f-calls", "g-calls", "CPU time", "Iterations"]
    plts = Vector{Plots.Plot}(length(pltlabels))

    cprdf = makefinite(rdf, maxfun)

    plts[1] = @df cprdf fun(:Solver, :fcalls, label=pltlabels[1])
    plts[2] = @df cprdf fun(:Solver, :gcalls, label=pltlabels[2])
    plts[3] = @df cprdf fun(:Solver, :CPUtime, label=pltlabels[3])
    plts[4] = @df cprdf fun(:Solver, :Iterations, label=pltlabels[4])
    return plot(plts...; yscale = yscale, kwargs...)
end


"""
Create performance profiles from a DataFrame of performance ratios.
Defaults to performance profiles of f-calls.
"""
function createperfprofiles(rdf::DataFrame, sym::Symbol = :fcalls;
                            t::Symbol = :steppost, xscale::Symbol = :log2,
                            ylims = (0,1),
                            maxfun = x -> 2*maximum(x), kwargs...)
    @assert sym in [:fcalls, :gcalls, :CPUtime, :Iterations]
    cprdf = makefinite(rdf, maxfun)
    plt = if sym == :fcalls
        @> cprdf begin
            @splitby _.Solver
            @x _.fcalls
            @y :cumulative
            @plot
        end
    elseif sym == :gcalls
        @> cprdf begin
            @splitby _.Solver
            @x _.gcalls
            @y :cumulative
            @plot
        end
    elseif sym == :CPUtime
        @> cprdf begin
            @splitby _.Solver
            @x _.CPUtime
            @y :cumulative
            @plot
        end
    elseif sym == :Iterations
        @> cprdf begin
            @splitby _.Solver
            @x _.Iterations
            @y :cumulative
            @plot
        end
    end
    return plot(plt; t=t, xscale=xscale, ylims=ylims, kwargs...)
end

"""
Create performance profiles from a vector of OptimizationRuns.
Defaults to performance profiles of f-calls.
"""
function createperfprofiles(oruns::Vector{OptimizationRun}, sym::Symbol = :fcalls,
                            solvers::AbstractVector{<:AbstractString} = Vector{String}(0);
                            t::Symbol = :steppost, xscale::Symbol = :log2,
                            ylims=(0,1),
                            kwargs...)
    createperfprofiles(createratiodataframe(oruns, solvers), sym;
                       t=t, xscale=xscale, ylims=ylims, kwargs...)
end
