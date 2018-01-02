@everywhere using AccelerationBenchmark.OACCEL2017

tests = [(:A,100), (:A,200),
         (:B,100), (:B,200),
         (:C,100), (:C,200),
         (:D,500), (:D,1000),
         (:E,100), (:E,200),
         (:F,200), (:F,500),
         (:G,100), (:G,200),
         (:D, 50000), (:D, 100000),
         (:E, 50000), (:E, 100000)]

seeds = 0:999
OACCEL2017.runmany(tests,seeds)
