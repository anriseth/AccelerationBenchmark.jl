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

function optim_problem(op::OptimizationProblem,
                       F = real(zero(eltype(initial_x(op)))),
                       G = similar(initial_x(op)),
                       H = spzeros(eltype(initial_x(op)), size(initial_x(op))...),
                       ) # TODO: How do we choose sparse vs full for H?
    if op.istwicedifferentiable
        df = TwiceDifferentiable(objective(op), gradient(op),
                                 UnconstrainedProblems.objective_gradient(op),
                                 UnconstrainedProblems.hessian(op),
                                 initial_x(op),
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

"""
Look up a the best (known) minimum of a CUTEst model.

Returns (minimizer, minimum)
"""
function minimumlookup(nlp::CUTEstModel)
    # TODO: Create approximate minimum
    # TODO: Should we store the minimizer, or just give it back a NaN?
    # TODO: Should the NaN-minimizer be of size nlp.meta.nvar?
    return [NaN], NaN
end

function optimizationproblem(nlp::CUTEstModel)
    # TODO: Set up database of minimum values for CUTEst problems
    # It seems like all the non-constrained CUTEst problems are twice differentiable?
    if nlp.meta.ncon > 0
        warn("We currently only support unconstrained CUTEst models, but $(nlp.meta.name) has `ncon > 0`.")
    end
    minimizer, minimum = minimumlookup(nlp)
    op = OptimizationProblem("$(nlp.meta.name)-$(nlp.meta.nvar)",
                             x->obj(nlp, x),
                             (g, x)->copy!(g, grad(nlp, x)),
                             (g, x)->cutest_fg!(nlp, x, g),
                             (h, x)->cutest_hess!(nlp, x, h),
                             nlp.meta.x0,
                             minimizer,
                             minimum,
                             true,
                             true)
end
