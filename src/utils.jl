function getopts(gt::GradientTolerance{T}, maxiter::Int) where T
    opts = Optim.Options(f_tol = zero(T), x_tol = zero(T),
                         g_tol = gt.g0norm*gt.tol,
                         allow_f_increases = true,
                         iterations = maxiter)
    return opts
end

function getopts(ft::FunctionTolerance{T}, maxiter::Int) where T
    cb = FunctionStop(ft.fL + ft.tol*(ft.f0 - ft.fL))
    opts = Optim.Options(f_tol = zero(T), x_tol = zero(T), g_tol = zero(T),
                         allow_f_increases = true,
                         iterations = maxiter, callback = cb)
    return opts
end

function runproblem(df, x0, solver::Optim.Optimizer, solvername::AbstractString,
                    problemname::AbstractString,
                    metrictol::StopTolerance;
                    recordtime::Bool = false, maxiter::Int = 1000)
    opts = getopts(metrictol, maxiter)

    f0 = value_gradient!(df, x0)
    g0norm = norm(gradient(df), Inf)

    r = optimize(df, x0, solver, opts)

    if recordtime
        # TODO: Do this with BenchmarkTools?
        runtime = @elapsed optimize(df, x0, solver, opts)
        runtime = min(runtime, @elapsed optimize(df, x0, solver, opts)) # Slight improvement on garbage collection?
    else
        runtime = NaN
    end
    #runtime = recordtime ? (@elapsed optimize(df, x0, solver, opts)) : NaN

    sname = isempty(solvername) ? summary(solver) : solvername
    OptimizationRun(Optim.iterations(r), Optim.f_calls(r), Optim.g_calls(r),
                    Optim.h_calls(r), runtime, Optim.minimum(r), f0,
                    Optim.g_residual(r), g0norm, solvername, problemname,
                    metrictol)
end

"""
Loop over solvers for problem `prob` and optimize until a given `stoptolerance`.

Return vector of `OptimizationRun`s for a given problem.
Optionally include runtime (runs optimization twice per solver).
"""
function createruns(prob, problemname::AbstractString,
                    solvers::AbstractVector, solvernames::AbstractVector{String},
                    tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Bool = false,
                    maxiter::Int = 1000) where T <: Real
    oruns = Vector{OptimizationRun{Float64,typeof(f0)}}(length(solvers))
    createruns!(oruns, prob, problemname, solvers, solvernames,
                tol, stoptype, timelog, maxiter)
    oruns
end

function createruns!(oruns, prob, problemname::AbstractString,
                     solvers::AbstractVector, solvernames::AbstractVector{String},
                     tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Bool = false,
                     maxiter::Int = 1000) where T <: Real
    # TODO: Is it problematic for specialized compilation to use stoptype::Symbol?
    # TODO: Update for twice differentiable problems
    # TODO: Update with fg! functionality
    x0 = initial_x(prob)
    df0 = optim_problem(prob)
    f0 = value_gradient!(df0, x0)
    g0norm = norm(gradient(df0), Inf)
    if stoptype == :GradientTolRelative
        metrictol = GradientTolerance(tol,g0norm)
    elseif stoptype == :GradientTol
        metrictol = GradientTolerance(tol,one(T))
    elseif stoptype == :FunctionTolRelative
        fL = solver_optimum(prob)
        @assert isfinite(fL)
        metrictol = FunctionTolerance(tol, f0, fL)
    elseif stoptype == :FunctionTol
        fL = solver_optimum(prob)
        @assert isfinite(fL)
        metrictol = FunctionTolerance(tol, one(fL) + fL, fL)
    else
        Base.error("The parameter `stoptype` must be one of :GradientTolRelative, :GradientTol, :FunctionTolRelative, or :FunctionTol.")
    end

    for (k, solver) in enumerate(solvers)
        df = optim_problem(prob)
        oruns[k] = runproblem(df, initial_x(prob), solver, solvernames[k],
                              problemname, metrictol;
                              recordtime=timelog, maxiter = maxiter)
    end

    return oruns
end

"""
Create performance measures of a list of solvers on a problem set.

Returns a vector of `OptimizationRun`s.
"""
function createmeasures(problems::AbstractVector, problemnames::AbstractVector{String},
                        solvers::AbstractVector, solvernames::AbstractVector{String},
                        tol::T, stoptype::Symbol = :GradientTolRelative, timelog::Bool = false,
                        maxiter::Int = 1000) where T <: Real
    np = length(problems); ns = length(solvers)
    oruns = Vector{OptimizationRun{Float64,T}}(np*ns)

    for (k,prob) in problems
        createruns!(view(oruns,(k-1)*ns+1:k*ns), prob, problemnames[k],
                    solvers, solvernames,
                    tol, stoptype, timelog, maxiter)
    end

    return oruns
end


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
    retval = x / minimum(x)
    retval[.!z .| isnan.(retval) .| isna.(retval)] = Inf # Should we use NA here?
    return retval
end

"Create a DataFrame of performance ratios from a dataframe of performance measures."
function createratiodataframe(dframe::DataFrame)
    by(dframe, :Problem) do dfr
        DataFrame(Solver = dfr[:Solver], rfcalls = _nanratio(dfr[:fcalls], dfr[:Success]),
                  rgcalls = _nanratio(dfr[:gcalls], dfr[:Success]),
                  rhcalls = _nanratio(dfr[:hcalls], dfr[:Success]),
                  rCPUtime = _nanratio(dfr[:CPUtime], dfr[:Success]),
                  Success = dfr[:Success])
    end
end

"Create a data frame with performance ratios from a vector of `OptimizationRun`s"
function createratiodataframe(oruns::Vector{OptimizationRun{T,Tf}}) where T where Tf
    createratiodataframe(createmeasuredataframe(oruns))
end
