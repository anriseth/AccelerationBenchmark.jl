module TestBench

using OptimTestProblems.UnconstrainedProblems
using AccelerationBenchmark
using Optim, LineSearches
using JLD, CSV

function defaulttestsetup()
    # Line search parameters taken from De Sterck
    # MATLAB files for Poblano toolbox.
    # The chosen gtol is *very* small. Typical values are 1e-1 for CG methods and 0.9 for quasi-Newton
    ls = MoreThuente(f_tol = 1e-4, gtol = 1e-2, x_tol = 1e-15,
                     stpmin = 1e-15, stpmax = 1e15, maxfev = 20)
    lsstatic = Static(alpha = 1e-4, scaled = true)
    ag = InitialStatic(alpha=1.0, scaled=false)

    gdls = GradientDescent(alphaguess = ag, linesearch = ls)
    gdst = GradientDescent(alphaguess = ag, linesearch = lsstatic)

    solvers = [
        LBFGS(alphaguess = ag, linesearch = ls, m = 5, scaleinvH0=true),
        NGMRES(alphaguess = ag, linesearch = ls, wmax = 20, precon = gdls),
        NGMRES(alphaguess = ag, linesearch = ls, wmax = 20, precon = gdst),
        OACCEL(alphaguess = ag, linesearch = ls, wmax = 20, precon = gdls),
        OACCEL(alphaguess = ag, linesearch = ls, wmax = 20, precon = gdst),
    ]
    solvernames = ["L-BFGS",
                   "N-GMRES-A", "N-GMRES-B",
                   "O-ACCEL-A", "O-ACCEL-B",
                   ]
    stoptype = :FunctionTolRelative
    stoptol  = 1e-10
    #stoptype = :GradientTolRelative
    #stoptol  = 1e-8
    timelog  = false
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

"Problem A in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createA(N::Int, seednum::Int)
    probname = "A-$N-$seednum"
    prob = UnconstrainedProblems._quadraticproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem B in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createB(N::Int, seednum::Int)
    probname = "B-$N-$seednum"
    prob = UnconstrainedProblems._paraboloidproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem C in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createC(N::Int, seednum::Int)
    probname = "C-$N-$seednum"
    prob = UnconstrainedProblems._paraboloidproblem(N; mat=eye(N))

    rnd = RandomizeInitialxMat([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem D in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createD(N::Int, seednum::Int)
    probname = "D-$N-$seednum"
    prob = UnconstrainedProblems._extrosenbrockproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem E in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createE(N::Int, seednum::Int)
    probname = "E-$N-$seednum"
    prob = UnconstrainedProblems._extpowellproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"""
Problem F in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017.

WARNING: `_trigonometricproblem` follows Moré et al. - *Testing unconstrained optimization software*, 1981,
which has a sign difference compared to Riseth.
"""
function createF(N::Int, seednum::Int)
    probname = "F-$N-$seednum"
    prob = UnconstrainedProblems._trigonometricproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"""
Problem G in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017.

WARNING: This function does not have a closed-form minimum value,
and the minimum depends on `N`.
"""
function createG(N::Int, seednum::Int)
    probname = "G-$N-$seednum"
    prob = UnconstrainedProblems._penfunIproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

macro createproblem(fun, N, seednum)
    eval(Symbol(:create,fun))(eval(N),eval(seednum))
end

"""
Run the same problem family multiple times in parallel with different randomized conditions.

Return a vector or `OptimizationRun`s
"""
function createrunsrandomizedparallel(fun::Symbol, N::Int, seeds::AbstractArray{Int},
                                      ts::TestSetup = defaulttestsetup())
    cfun = eval(Symbol(:create,fun))
    np = length(seeds)
    ns = length(ts.solvers)
    oruns = Vector{OptimizationRun{Float64,Float64}}(np*ns)
    for k = 1:np
        prob, probname = cfun(N,seeds[k])
        @show probname
        createruns!(view(oruns,(k-1)*ns+1:k*ns), prob, probname,
                    ts.solvers, ts.solvernames,
                    ts.stoptol, ts.stoptype, ts.timelog, ts.maxiter)
    end
    return oruns
end

"""
Run problems and create tables with the corresponding measures, to be used
to investigate performance and plot performance profiles.
"""
function runmany(funs::AbstractVector, seeds::AbstractArray{Int},
                 ts::TestSetup = defaulttestsetup();
                 savejld::Bool = true,
                 savecsv::Bool = true,
                 savebase::AbstractString = "data/")
    # funs is a collection of Tuple{Symbol,Int}
    @assert savejld || savecsv
    for fN in funs
        oruns = createrunsrandomizedparallel(fN..., seeds, ts)
        mdf = createmeasuredataframe(oruns)
        if savejld
            save(savebase*"$(fN[1])-$(fN[2]).jld", "mdf", mdf)
        end
        if savecsv
            CSV.write(savebase*"$(fN[1])-$(fN[2]).csv", mdf)
        end
    end
end

end
