#=
This code can be used to replicate the test results from
A.N. Riseth - "Objective acceleration for unconstrained optimization", 2017.
https://arxiv.org/abs/1710.05200
=#
module OACCEL2017

using OptimTestProblems.MultivariateProblems
using AccelerationBenchmark
using Optim, LineSearches
using JLD, CSV

const MVP = MultivariateProblems
const UP = MVP.UnconstrainedProblems

function defaulttestsetup()
    # Line search parameters taken from De Sterck
    # MATLAB files for Poblano toolbox.
    # The De Sterck gtol=1e-2 is *very* small. Typical values are 1e-1 for CG methods and 0.9 for quasi-Newton
    # In the updated paper we use gtol=0.1.
    ls = MoreThuente(f_tol = 1e-4, gtol = 1e-1, x_tol = 1e-15,
                     alphamin = 1e-15, alphamax = 1e15, maxfev = 20)
    lsstatic = Static()
    agscaled = InitialStatic(alpha = 1e-4, scaled = true)
    ag = InitialStatic(alpha=1.0, scaled=false)

    gdls = GradientDescent(alphaguess = ag, linesearch = ls)
    gdst = GradientDescent(alphaguess = agscaled, linesearch = lsstatic)

    solvers = [
        LBFGS(alphaguess = ag, linesearch = ls, m = 5, scaleinvH0=true),
        NGMRES(alphaguess = ag, linesearch = ls, wmax = 20, nlprecon = gdls),
        NGMRES(alphaguess = ag, linesearch = ls, wmax = 20, nlprecon = gdst),
        OACCEL(alphaguess = ag, linesearch = ls, wmax = 20, nlprecon = gdls),
        OACCEL(alphaguess = ag, linesearch = ls, wmax = 20, nlprecon = gdst),
    ]
    solvernames = ["L-BFGS",
                   "N-GMRES-A", "N-GMRES-B",
                   "O-ACCEL-A", "O-ACCEL-B",
                   ]
    stoptype = :FunctionTolRelative
    stoptol  = 1e-10
    #stoptype = :GradientTolRelative
    #stoptol  = 1e-8
    timelog  = 0 # timelog = 1 can cause very inaccurate timings due to compilation and garbage collection
    maxiter  = 2000
    timelimit = NaN
    TestSetup(solvers,solvernames,stoptype,stoptol,timelog,maxiter,timelimit)
end


"Problem A in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createA(N::Int, seednum::Int)
    probname = "A-$N-$seednum"
    prob = UP._quadraticproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem B in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createB(N::Int, seednum::Int)
    probname = "B-$N-$seednum"
    prob = UP._paraboloidproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem C in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createC(N::Int, seednum::Int)
    probname = "C-$N-$seednum"
    prob = UP._paraboloidproblem(N; mat=eye(N))

    rnd = RandomizeInitialxMat([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem D in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createD(N::Int, seednum::Int)
    probname = "D-$N-$seednum"
    prob = UP._extrosenbrockproblem(N)

    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

"Problem E in  AN Riseth - *Objective acceleration for unconstrained optimization*, 2017."
function createE(N::Int, seednum::Int)
    probname = "E-$N-$seednum"
    prob = UP._extpowellproblem(N)

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
    prob = UP._trigonometricproblem(N)

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
    prob = UP._penfunIproblem(N)
    rnd = RandomizeInitialx([seednum])
    randomizeproblem!(prob, rnd, 1)
    return prob, probname
end

macro createproblem(fun, N, seednum)
    eval(Symbol(:create,fun))(eval(N),eval(seednum))
end

"Helper function for pmap in createrunsrandomized."
function getoruns(seed::Int, N::Int, cfun, ts::TestSetup, showname::Bool)
    prob, probname = cfun(N,seed)
    showname && println("$probname")
    createruns(prob, probname,
               ts.solvers, ts.solvernames,
               ts.stoptol, ts.stoptype, ts.timelog, ts.maxiter;
               time_limit = ts.timelimit)
end

"""
Run the same problem family multiple times with different randomized conditions.

Return a vector or `OptimizationRun`s
"""
function createrunsrandomized(fun::Symbol, N::Int, seeds::AbstractArray,
                              ts::TestSetup = defaulttestsetup();
                              showname::Bool = false)
    cfun = eval(Symbol(:create,fun))
    np = length(seeds)

    oruns = reduce(append!, pmap(k->getoruns(seeds[k],N,cfun,ts,showname), 1:np))

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
                 savebase::AbstractString = "data/",
                 showname::Bool = true)
    # funs is a collection of Tuple{Symbol,Int}
    @assert savejld || savecsv
    if myid() == 1 && !isdir(savebase)
        mkdir(savebase)
    end
    for fN in funs
        oruns = createrunsrandomized(fN..., seeds, ts; showname=showname)
        mdf = createmeasuredataframe(oruns)
        if savejld
            myid() == 1 && save(savebase*"/$(fN[1])-$(fN[2]).jld", "mdf", mdf)
        end
        if savecsv
            myid() == 1 && CSV.write(savebase*"/$(fN[1])-$(fN[2]).csv", mdf)
        end
    end
end

end
