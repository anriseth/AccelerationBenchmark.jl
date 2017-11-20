module TestBench

using OptimTestProblems.UnconstrainedProblems
using AccelerationBenchmark
using Optim, LineSearches


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

# TODO: make a macro that generates each of the run[A-G] functions below.
# They are all the same, except the probnamebase letter, and that runC uses RandomInitialxMat.

"Problem A in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function runA(seeds::AbstractArray, N::Int, ts::TestSetup = defaulttestsetup())
    rnd = RandomizeInitialx(seeds)
    probnamebase = "A-$N"
    prob = UnconstrainedProblems._quadraticproblem(N)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

"Problem B in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function runB(seeds::AbstractArray, N::Int, ts::TestSetup =  defaulttestsetup())
    rnd = RandomizeInitialx(seeds)
    probnamebase = "B-$N"
    prob = UnconstrainedProblems._paraboloidproblem(N)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

"Problem C in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function runC(seeds::AbstractArray, N::Int, ts::TestSetup =  defaulttestsetup())
    rnd = RandomizeInitialxMat(seeds)
    probnamebase = "C-$N"
    prob = UnconstrainedProblems._paraboloidproblem(N; mat=eye(N))

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

"Problem D in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function runD(seeds::AbstractArray, N::Int, ts::TestSetup = defaulttestsetup())
    rnd = RandomizeInitialx(seeds)
    probnamebase = "D-$N"
    prob = UnconstrainedProblems._extrosenbrockproblem(N)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

"Problem E in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function runE(seeds::AbstractArray, N::Int, ts::TestSetup = defaulttestsetup())
    rnd = RandomizeInitialx(seeds)
    probnamebase = "E-$N"
    prob = UnconstrainedProblems._extpowellproblem(N)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end


"""
Problem F in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017.

WARNING: `_trigonometricproblem` follows Moré et al. - *Testing unconstrained optimization software*, 1981,
which has a sign difference compared to Riseth.
"""
function runF(seeds::AbstractArray, N::Int, ts::TestSetup = defaulttestsetup())
    rnd = RandomizeInitialx(seeds)
    probnamebase = "F-$N"
    prob = UnconstrainedProblems._trigonometricproblem(N)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

"""
Problem G in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017.

WARNING: This function does not have a closed-form minimum value,
and the minimum depends on `N`.
"""
function runG(seeds::AbstractArray, N::Int, ts::TestSetup = defaulttestsetup())
    rnd = RandomizeInitialx(seeds)
    probnamebase = "G-$N"
    prob = UnconstrainedProblems._penfunIproblem(N)

    oruns = createrunsrandomized(prob, probnamebase, rnd, ts)
end

end
