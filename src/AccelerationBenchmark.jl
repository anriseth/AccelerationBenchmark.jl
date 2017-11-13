#=
TODO:
-
=#

module AccelerationBenchmark

using OptimTestProblems.UnconstrainedProblems
using NLSolversBase
using Optim, LineSearches


" Stored results of solvers for a given optimization problem "
immutable OptimizationReport{T,Tf}
    f0::T
    g0norm::T
    oruns::Vector{OptimizationRun{T,Tf}}
end

" Log of results from a given optimization run "
immutable OptimizationRun{T,Tf}
    iterations::Int
    fevals::Int
    gevals::Int
    hevals::Int
    cputime::T
    fval::Tf
    f0::Tf
    gnorm::Tf
    g0norm::Tf
    solvername::AbstractString
    success::Bool
end

function OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval,
                         f0, gnorm, g0norm, solvername, ft::FunctionTolerance)
    success = fval ≤ ft.fL + ft.tol*(ft.f0 - ft.fL)
    OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval, f0,
                    gnorm, gnorm, solvername, success)
end


function OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval,
                         gnorm, solvername, gt::GradientTolerance)
    success = gnorm ≤ gt.tol*gt.g0norm
    OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval, gnorm,
                    solvername, success)
end

"""
Callback to stop optimization if the function value is below some threshold.
"""
immutable FunctionStop{T}
    val::T
end

function (cb::FunctionStop)(os::OptimizationState)
    os.value ≤ cb.val
end

function (cb::FunctionStop)(tr::OptimizationTrace)
    tr[end].value ≤ cb.val
end


abstract type StopTolerance end;

"""
Tell the optimization procedure to exit when we hit `|g(x)|≤ tol * |g(x_0)|`.

For absolute, rather than relative, tolerance, set `g0norm = 1.0`.
"""
immutable GradientTolerance{T} <: StopTolerance
    # TODO: The g0norm is a bit superfluous
    tol::T
    g0norm::T
end

"""
Tell the optimization procedure to exit when we hit `f(x) ≤ f_L + tol * (f(x_0) - f_L)`,
where `f_L` is the minimum value of `f`.

For absolute, rather than relative, tolerance, set `f0 = 1.0 + fL`.
"""
immutable FunctionTolerance{T} <: StopTolerance
    # TODO: f0 and fL are superflous
    tol::T
    f0::T
    fL::T
end

"""
If we don't know the value, tell the optimization procedure that we want to find
the minimum function value achieved across a list of solvers,
by running them all to within a given gradient norm tolerance and number of iterations.
"""
immutable UnknownFunctionTolerance{T} <: StopTolerance
    # TODO: Maybe we just run these problems to high tolerance before
    #       starting this process, and then store the approximate minimum value.
    gtol::T
    iterations::Int
end

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
                    Optim.g_residual(r), g0norm, metrictol)
end

"""
Loop over solvers for problem `prob` and optimize until a given stoptolerance.

Return OptimizationReport with performance metrics.
Optionally include runtime (runs optimization twice per solver).
"""
function createreport(prob, solvers::AbstractVector, solvernames::AbstractVector{AbstractString},
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

    oruns = Vector{OptimizationReport{Float64,typeof(f0)}}(length(solvers))
    for (k, solver) in enumerate(solvers)
        df = OnceDifferentiable(objective(prob), gradient(prob),
                                copy(prob.initial_x)) # Copy just in case
        oruns[k] = runproblem(df, prob.initial_x, solver, metrictol;
                              recordtime=timelog, maxiter = maxiter)
    end

    return oruns
end


end # module
