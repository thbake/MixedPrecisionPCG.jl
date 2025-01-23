export ConvergenceData
export compute_error_norm!, compute_error!, A_norm, A_norm!, compute_residual_norm!, 
       compute_backward_error!, preconditioned_condition_number

mutable struct ConvergenceData{T}

    n                      ::Int          # Number of dimension.
    iter_number            ::Int          # Number of iterations.
    residual_norm          ::Vector{T}    # Relative residual norm per iteration. 
    relative_error_norm    ::Vector{T}    # Relative error in the A-norm.
    relative_backward_error::Vector{T}    # Relative backward error.
    iterates              ::Matrix{T}    # n by k matrix representing the k iterates.

    function ConvergenceData{T}(n::Int, max_iter::Int) where T<:AbstractFloat

        new(n, max_iter, ones(max_iter), ones(max_iter), ones(max_iter), ones(n, max_iter))

    end

end

"""
Given a symmetric/Hermitian positive definite (SPD/HPD) matrix A and a vector x 
compute the error in the A-norm.
"""
A_norm(A, x::AbstractVector) = sqrt(dot(x, A*x))

"""
Given a ConvergenceData struct, an SPD/HPD matrix A and a collection of k 
vectors in the form of an n by k matrix X, compute the error in the A-norm.
"""
function A_norm!(cd::ConvergenceData, A, X::AbstractMatrix) 

    M = A * X

    cd.relative_error_norm = [

        sqrt( 

            dot(@view(X[:, k]), @view(M[:, k]))

        ) for k in 1:cd.iter_number
    ]

end

""" 
Compute error in the A-norm 
"""
function compute_error_norm!(
    cd::ConvergenceData,
    exact_solution  ::Vector,
    A,
    ::FactorizationPreconditioner) 

    # This should be an (n x k) error matrix.
    error_matrix = exact_solution .- cd.iterates

    # Update error.
    A_norm!(cd, A, error_matrix)

    cd.relative_error_norm .* inv(A_norm(A, exact_solution))

end

function compute_error!(cd::ConvergenceData, exact_solution::AbstractVector, A)

    errors        = ones(size(A, 1), cd.iter_number)
    solution_norm = A_norm(A, exact_solution)

    for k in 1:cd.iter_number

        errors[:, k] = cd.iterates[:, k] - exact_solution
        tmp          = A * errors[:, k]
        cd.relative_error_norm[k] = sqrt(dot(tmp, @view(errors[:, k]))) / solution_norm

    end
end

function compute_error!(
    cd            ::ConvergenceData,
    exact_solution::AbstractVector,
    A,
    M::FactorizationPreconditioner{uL, uR, Right}) where {uL, uR} 

    errors         = ones(size(A, 1), cd.iter_number)
    solution_norm  = A_norm(A, exact_solution)

    for k in 1:cd.iter_number

        errors[:, k] = precondition(M, cd.iterates[:, k] - exact_solution)
        tmp          = A * errors[:, k]
        cd.relative_error_norm[k] = sqrt(dot(tmp, @view(errors[:, k]))) / solution_norm

    end
end

"""
Compute residual norm.
"""
function compute_residual_norm!(cd::ConvergenceData, A, b)

    xk     = cd.iterates

    for k in 1:cd.iter_number

        cd.residual_norm[k] = norm(b - A * @view(xk[:, k])) 

    end
end

"""
Compute normwise backward error.
"""

function compute_backward_error!(cd::ConvergenceData, A, b)

    compute_residual_norm!(cd, A, b)

    Anorm = norm(A)
    bnorm = norm(b)

    for k in 1:cd.iter_number

        approxnorm                    = norm(@view cd.iterates[:, k])
        cd.relative_backward_error[k] = cd.residual_norm[k] * inv( (Anorm * approxnorm) + bnorm )

   end

end

function compute_backward_error!(
    cd::ConvergenceData,
    A,
    M::FactorizationPreconditioner{uL, uR, Right},
    b) where {uL, uR, Right}

    compute_residual_norm!(cd, A, b)

    Anorm = norm(A)
    bnorm = norm(b)

    for k in 1:cd.iter_number

        approxnorm                    = norm(precondition(M, @view cd.iterates[:, k]))
        cd.relative_backward_error[k] = cd.residual_norm[k] * inv((Anorm * approxnorm) + bnorm)

    end
    

end

function preconditioned_condition_number(
    A::AbstractMatrix,
    M::FactorizationPreconditioner{uL, uR, T}) where {uL, uR, T<:PreconditioningScheme}

    Aprec = precondition(M, A)

    return cond(Aprec)

end

