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


"Log of results from a given optimization run."
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
    solvername::String
    problemname::String
    success::Bool
end

function OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval,
                         f0, gnorm, g0norm, solvername, problemname,
                         ft::FunctionTolerance)
    success = fval ≤ ft.fL + ft.tol*(ft.f0 - ft.fL)
    OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval, f0,
                    gnorm, g0norm, solvername, problemname, success)
end


function OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval,
                         f0, gnorm, g0norm, solvername, problemname,
                         gt::GradientTolerance)
    success = gnorm ≤ gt.tol*gt.g0norm
    OptimizationRun(iterations, fevals, gevals, hevals, cputime, fval, f0,
                    gnorm, g0norm, solvername, problemname, success)
end


"Collection of options used create performance measures."
immutable TestSetup{T<:Real, Ts<:AbstractVector, Tn<:AbstractVector, Tl<:Real}
    solvers::Ts
    solvernames::Tn
    stoptype::Symbol
    stoptol::T
    timelog::Bool
    maxiter::Int
    timelimit::Tl # Soft time limit for each optimize call
end
