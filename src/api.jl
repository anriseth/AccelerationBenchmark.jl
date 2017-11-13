import OptimTests: initial_x, solution_optimum, optim_problem

initial_x(p::OptimizationProblem) = p.initial_x
solver_optimum(p::OptimizationProblem) = p.minimum

function optim_problem(op::OptimizationProblem)
    if op.istwicedifferentiable
        df = TwiceDifferentiable(objective(op), gradient(op),
                                 hessian(op), initial_x(op))
    elseif op.isdifferentiable
        df = OnceDifferentiable(objective(op), gradient(op), initial_x(op))
    else
        error("Only implemented for differentiable problems.")
    end
end
