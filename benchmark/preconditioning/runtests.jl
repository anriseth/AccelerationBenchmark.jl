@everywhere using AccelerationBenchmark
using DataFrames, CSV, Optim, LineSearches, CUTEst, JLD
run_tests_more = true # Run the Moré et al. tests from the O-ACCEL paper
run_tests_cutest = true # Run CUTEst tests

savejld = true
savecsv = true

function gettestset()
    wmax = 20
    m = 10
    ag = InitialStatic(alpha=1.0, scaled=false)

    # delta and sigma are the Wolfe condition coefficients
    lshz = HagerZhang(delta = 0.1, sigma = 0.9) # Defaults
    # lslbfgs = HagerZhang(delta = 1e-4, sigma = 0.9)
    # lsgd = HagerZhang(delta = 1e-4, sigma = 0.1)
    # lsacc = HagerZhang(delta = 1e-4, sigma = 0.5)
    lslbfgs = lshz
    lsgd = lshz
    lsacc = lshz

    preconls = Static(alpha=1e-4, scaled=true)
    gdprecon = GradientDescent(linesearch = preconls,
                               alphaguess = ag) # alphaguess unnecessary?

    solvers = [
        LBFGS(alphaguess = ag, linesearch = lslbfgs, scaleinvH0 = true, m = m),
        OACCEL(alphaguess = ag, linesearch = lsacc,
               precon = gdprecon, wmax = wmax),
        NGMRES(alphaguess = ag, linesearch = lsacc,
               precon = gdprecon, wmax = wmax),
        OACCEL(alphaguess = ag, linesearch = lsacc,
               precon = gdprecon, wmax = m),
        NGMRES(alphaguess = ag, linesearch = lsacc,
               precon = gdprecon, wmax = m),
        GradientDescent(alphaguess = ag, linesearch = lsgd)
    ]
    solvernames = [
        "L-BFGS",
        "O-ACCEL-$wmax",
        "N-GMRES-$wmax",
        "O-ACCEL-$m",
        "N-GMRES-$m",
        "GD"
    ]
    #stoptype = :FunctionTolRelative
    #stoptol  = 1e-10
    stoptype = :GradientTolRelative
    stoptol  = 1e-8
    timelog  = 2 # timelog = 1 can cause very inaccurate timings due to compilation and garbage collection
    maxiter  = 10000
    timelimit = 65.0
    TestSetup(solvers,solvernames,stoptype,stoptol,timelog,maxiter,timelimit)
end

testset = gettestset()
savebase = "data/"

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

    OACCEL2017.runmany(tests,seeds, testset;
                       savejld = savejld, savecsv = savecsv,
                       savebase = savebase)
end

if run_tests_cutest
    min_var = 20
    max_var = 100000
    cutestnames = sort(CUTEst.select(contype=:unc, max_var=max_var,
                                     min_var=min_var))
    oruns = createmeasures(cutestnames, testset)
    mdf = createmeasuredataframe(oruns)

    if savejld
        save(savebase*"/cutest_$(min_var)_$(max_var).jld", "mdf", mdf)
    end
    if savecsv
        CSV.write(savebase*"/cutest_$(min_var)_$(max_var).csv", mdf)
    end
end
