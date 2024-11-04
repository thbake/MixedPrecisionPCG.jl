export ConvergenceData
export compute_error_norm!, A_norm, A_norm!, compute_residual_norm!, compute_backward_error!

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

function compute_error_norm!(
    cd::ConvergenceData,
    exact_solution  ::Vector,
    A)

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

function compute_residual_norm!(cd::ConvergenceData, A, b)

    xk     = cd.iterates

    for k in 1:cd.iter_number

        cd.residual_norm[k] = norm(b - A * @view(xk[:, k])) 

    end
end

function compute_backward_error!(cd::ConvergenceData, A, b)

    compute_residual_norm!(cd, A, b)

    Anorm = norm(A)
    bnorm = norm(b)

    for k in 1:cd.iter_number

        approxnorm                    = norm(@view cd.iterates[:, k])
        cd.relative_backward_error[k] = cd.residual_norm[k] * inv( (Anorm * approxnorm) + bnorm )

   end

end
