import OptimTests: initial_x, solution_optimum, optim_problem

initial_x(p::OptimizationProblem) = p.initial_x
solver_optimum(p::OptimizationProblem) = p.minimum

function cutest_fg!(nlp, x, g)
    grad!(nlp, x, g)
    obj(nlp, x)
end

function symmetrize!(h)
    for j = 1:size(h,2)
        for i = 1:j-1
            h[i,j] = h[j,i]
        end
    end
    h
end

function cutest_hess!(nlp, x, h)
    symmetrize!(copy!(h, hess(nlp, x)))
end

"""
    initial_x(nlp) -> x0

Return the starting point for an optimization problem `nlp`.
"""
initial_x(nlp::CUTEstModel) = nlp.meta.x0

function optim_problem(op::OptimizationProblem,
                       F = real(zero(eltype(initial_x(op)))),
                       G = similar(initial_x(op)),
                       H = NLSolversBase.alloc_H(initial_x(op)))
    if op.istwicedifferentiable
        df = TwiceDifferentiable(objective(op), gradient(op),
                                 UnconstrainedProblems.objective_gradient(op),
                                 UnconstrainedProblems.hessian(op),
                                 initial_x(op)
                                 F, G, H)
    elseif op.isdifferentiable
        df = OnceDifferentiable(objective(op), gradient(op),
                                UnconstrainedProblems.objective_gradient(op),
                                initial_x(op),
                                F, G)
    else
        error("Only implemented for differentiable problems.")
    end

    return df
end

function optim_problem(nlp::CUTEstModel)
    # TODO: is there a way to check whether the hessian is available?
    # It seems like all the non-constrained CUTEst problems are twice differentiable?
    if nlp.meta.ncon > 0
        error("We currently only support unconstrained CUTEst models.")
    end

    # TODO: use sparse Hessian cache?
    d = TwiceDifferentiable(x -> obj(nlp, x),
                            (g, x) -> grad!(nlp, x, g),
                            (g, x) -> cutest_fg!(nlp, x, g),
                            (h, x) -> cutest_hess!(nlp, x, h),
                            initial_x(nlp))
    return d
end

"Returns the best-known minimum value of a CUTEst model stored"
function minimumlookup(nlp::CUTEstModel)
    return NaN
end

function optimizationproblem(nlp::CUTEstModel)
    # TODO: Set up database of minimum values for CUTEst problems
    # It seems like all the non-constrained CUTEst problems are twice differentiable?
    if nlp.meta.ncon > 0
        error("We currently only support unconstrained CUTEst models.")
    end
    minval = minimumlookup(nlp)
    op = OptimizationProblem("$(nlp.meta.name)-$(nlp.meta.nvar)",
                             x->obj(nlp, x),
                             (g, x)->copy!(g, grad(nlp, x)),
                             (g, x)->cutest_fg!(nlp, x, g),
                             (h, x)->cutest_hess!(nlp, x, h),
                             initial_x(nlp),
                             minval,
                             true,
                             true)
end
