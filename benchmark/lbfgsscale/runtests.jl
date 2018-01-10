using AccelerationBenchmark, DataFrames, CSV, Optim, LineSearches, CUTEst, JLD
const run_tests_more = true # Run the Moré et al. tests from the O-ACCEL paper
const run_tests_cutest = true # Run CUTEst tests

const savejld = true
const savecsv = true

function lstests()
    lsm = MoreThuente()
    lsh = HagerZhang()
    lsbt2 = BackTracking(order=2)
    lsbt3 = BackTracking(order=3)

    ag = InitialStatic(alpha=1.0, scaled=false)

    solvers = [
        LBFGS(alphaguess = ag, linesearch = lsm, scaleinvH0=true),
        LBFGS(alphaguess = ag, linesearch = lsm, scaleinvH0=false),
        LBFGS(alphaguess = ag, linesearch = lsh, scaleinvH0=true),
        LBFGS(alphaguess = ag, linesearch = lsh, scaleinvH0=false),
        LBFGS(alphaguess = ag, linesearch = lsbt2, scaleinvH0=true),
        LBFGS(alphaguess = ag, linesearch = lsbt2, scaleinvH0=false),
        LBFGS(alphaguess = ag, linesearch = lsbt3, scaleinvH0=true),
        LBFGS(alphaguess = ag, linesearch = lsbt3, scaleinvH0=false),
    ]
    solvernames = [
        "(MT,true)", "(MT,false)",
        "(HZ,true)", "(HZ,false)",
        "(BT2,true)", "(BT2,false)",
        "(BT3,true)", "(BT3,false)",
    ]
    #stoptype = :FunctionTolRelative
    #stoptol  = 1e-10
    stoptype = :GradientTolRelative
    stoptol  = 1e-10
    timelog  = true
    maxiter  = 5000
    time_limit = NaN
    TestSetup(solvers,solvernames,stoptype,stoptol,timelog,maxiter,time_limit)
end

if run_tests_more
    tests = [(:A,100), (:A,200),
             (:B,100), (:B,200),
             (:C,100), (:C,200),
             (:D,500), (:D,1000),
             (:E,100), (:E,200),
             (:F,200), (:F,500),
             (:G,100), (:G,200),
             (:D, 50000), (:D, 100000),
             (:E, 50000), (:E, 100000)]
    seeds = 0:99

    OACCEL2017.runmany(tests,seeds, lstests();
                       savejld = savejld, savecsv = savecsv,
                       savebase = "data/")
end


if run_tests_cutest
    min_var = 0
    max_var = 1000
    cutestnames = sort(CUTEst.select(contype=:unc, max_var=max_var,
                                     min_var=min_var))
    splice!(cutestnames, findfirst(cutestnames,"KOWOSBNE")) # This is actually not unconstrained
    oruns = createmeasures(cutestnames, lstests())
    mdf = createmeasuredataframe(oruns)

    if savejld
        save(savebase*"/cutest_$(min_var)_$(max_var).jld", "mdf", mdf)
    end
    if savecsv
        CSV.write(savebase*"/cutest_$(min_var)_$(max_var).csv", mdf)
    end
end
