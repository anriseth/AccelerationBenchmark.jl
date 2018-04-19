abstract type RandomizeProblem end

immutable RandomizeInitialx{T<:AbstractArray} <: RandomizeProblem
    seeds::T
end

immutable RandomizeInitialxMat{T<:AbstractArray} <: RandomizeProblem
    seeds::T
end

Base.length(r::RandomizeProblem) = length(r.seeds)

function randomizeproblem!(prob::OptimizationProblem, rnd::RandomizeInitialx, k::Int)
    srand(rnd.seeds[k])
    rand!(prob.initial_x)
end

function randomizeproblem!(prob::OptimizationProblem, rnd::RandomizeInitialxMat, k::Int)
    srand(rnd.seeds[k])
    rand!(prob.initial_x)
    prob.parameters.mat .= UP._randommatrix(length(prob.initial_x), true)
end
