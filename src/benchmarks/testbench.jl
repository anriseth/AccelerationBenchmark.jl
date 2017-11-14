module TestBench

using OptimTestProblems.UnconstrainedProblems
using AccelerationBenchmark
using Optim, LineSearches

immutable TestSetup{T<:Real, Ts<:AbstractVector, Tn<:AbstractVector}
    solvers::Ts
    solvernames::Tn
    stoptype::Symbol
    stoptol::T
    timelog::Bool
    maxiter::Int
end

function defaulttestsetup()
    solvers = [LBFGS(scaleinvH0=false,linesearch=BackTracking(order=3)),
               LBFGS(scaleinvH0=false,linesearch=BackTracking(order=2)),
               LBFGS(scaleinvH0=true,linesearch=BackTracking(order=3)),
               LBFGS(scaleinvH0=true,linesearch=BackTracking(order=2)),
               LBFGS(scaleinvH0=true,linesearch=MoreThuente()),
               LBFGS(scaleinvH0=false,linesearch=MoreThuente())]
    solvernames = ["LBFGS(false,BT3)", "LBFGS(false,BT2)",
                   "LBFGS(true,BT3)", "LBFGS(true,BT2)",
                   "LBFGS(true,MT)", "LBFGS(false,MT)"]
    stoptype = :GradientTolRelative
    stoptol  = 1e-8
    timelog  = true
    maxiter  = 1500
    TestSetup(solvers,solvernames,stoptype,stoptol,timelog,maxiter)
end


abstract type RandomizeProblem end

immutable RandomizeInitialx{T<:AbstractArray} <: RandomizeProblem
    seeds::T
end

immutable RandomizeInitialxMat{T<:AbstractArray} <: RandomizeProblem
    seeds::T
end

Base.length(r::RandomizeProblem) = length(r.seeds)

function randomizeproblem!(prob::OptimizationProblem, rnd::RandomizeInitialx)
    rand!(prob.initial_x)
end

function randomizeproblem!(prob::OptimizationProblem, rnd::RandomizeInitialxMat)
    rand!(prob.initial_x)
    prob.parameters.mat .= UnconstrainedProblems._randommatrix(length(prob.initial_x), true)
end

function createrunsrandomized(prob::OptimizationProblem, rnd::RandomizeProblem,
                              probnamebase::AbstractString,
                              solvers::AbstractVector, solvernames::AbstractVector{String},
                              tol::T, stoptype::Symbol = :GradientTolRelative,
                              timelog::Bool = false,
                              maxiter::Int = 1000) where T <: Real
    np = length(rnd)
    ns = length(solvers)
    oruns = Vector{OptimizationRun{Float64,Float64}}(np*ns)
    for (k, seed) in enumerate(rnd.seeds)
        srand(seed)
        randomizeproblem!(prob, rnd) # TODO: pass seed/k in here in here?
        createruns!(view(oruns,(k-1)*ns+1:k*ns), prob, probnamebase*"-$seed",
                    solvers, solvernames,
                    tol, stoptype, timelog, maxiter)
    end
    return oruns
end

function createrunsrandomized(prob::OptimizationProblem, probnamebase::AbstractString,
                              rnd::RandomizeProblem, ts::TestSetup)
    createrunsrandomized(prob, rnd, probnamebase, ts.solvers, ts.solvernames,
                         ts.stoptol, ts.stoptype, ts.timelog, ts.maxiter)
end



function runit()
    ts = defaulttestsetup()
    rnd = RandomizeInitialx(0:99)
    probnamebase = "A-100"
    prob = UnconstrainedProblems._quadraticproblem(100)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

end
