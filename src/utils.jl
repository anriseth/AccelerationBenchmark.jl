function getopts(gt::GradientTolerance{T}, maxiter::Int)
    opts = Optim.Options(f_tol = zero(T), x_tol = zero(T),
                         g_tol = gt.g0norm*gt.tol,
                         allow_f_increases = true,
                         iterations = maxiter)
    return opts
end

function getopts(ft::FunctionTolerance{T}, maxiter::Int)
    cb = FunctionStop(ft.fL + ft.tol*(ft.f0 - ft.fL))
    opts = Optim.Options(f_tol = zero(T), x_tol = zero(T), g_tol = zero(T),
                         allow_f_increases = true,
                         iterations = maxiter, callback = cb)
    return opts
end

function runproblem(df, x0, solver::Optimizer, solvername::AbstractString,
                    problemname::AbstractString,
                    metrictol::StopTolerance;
                    recordtime::Bool = false, maxiter::Int = 1000)
    opts = getopts(metrictol, maxiter)

    f0 = value_gradient!(df, x0)
    g0norm = norm(gradient(df), Inf)

    r = optimize(df, x0, solver, opts)

    runtime = recordtime ? @elapsed optimize(df, x0, solver, opts) : NaN

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
function createruns(prob, solvers::AbstractVector, solvernames::AbstractVector{AbstractString},
                    tol::T, stoptype::Symbol = GradientTolRelative, timelog::Bool = false,
                    maxiter::Bool = 1000) where T <: Real
    # TODO: Is it problematic for specialized compilation to use stoptype::Symbol?
    # TODO: Update for twice differentiable problems
    # TODO: Update with fg! functionality
    if stoptype == :GradientTolRelative
        metrictol = GradientTolerance(tol,g0norm)
    elseif stoptype == :GradientTol
        metrictol = GradientTolerance(tol,one(T))
    elseif stoptype == :FunctionTolRelative
        metrictol = FunctionTolerance(tol, f0, value(df0,prob.solutions))
    elseif stoptype == :FunctionTol
        fL = value(df0, prob.solutions)
        metrictol = FunctionTolerance(tol, one(fL) + fL, fL)
    else
        Base.error("The parameter `stoptype` must be one of :GradientTolRelative, :GradientTol, :FunctionTolRelative, or :FunctionTol.")
    end

    oruns = Vector{OptimizationRun{Float64,typeof(f0)}}(length(solvers))
    for (k, solver) in enumerate(solvers)
        df = OnceDifferentiable(objective(prob), gradient(prob),
                                copy(prob.initial_x)) # Copy just in case
        oruns[k] = runproblem(df, prob.initial_x, solver, metrictol,
                              solvernames[k], probname;
                              recordtime=timelog, maxiter = maxiter)
    end

    return oruns
end


"Take a vector of `OptimizationRun`s and convert it to a DataFrame"
function createdataframe(oruns::Vector{OptimizationRun{T,Tf}}) where T where Tf
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
        sucarr[k]    = orun.success
    end

    dframe = DataFrame([pnames, snames, iters, fcallarr, gcallarr, hcallarr, cputimes, fvalarr, f0arr, gnormarr, g0normarr, succarr]
                       [:Problem, :Solver, :Iterations, :fcalls, :gcalls, :hcalls, :CPUtime, :fval, :f0, :gnorm, :g0norm, :Success])
end
