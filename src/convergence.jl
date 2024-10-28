export ConvergenceData
export compute_error_norm!, A_norm, A_norm!, compute_residual_norm!

mutable struct ConvergenceData{T}

    n                     ::Int          # Number of dimension.
    iter_number           ::Int          # Number of iterations.
    relative_residual_norm::Vector{T}    # Relative residual norm per iteration. 
    relative_error_norm   ::Vector{T}    # Relative error in the A-norm.
    iterates              ::Matrix{T}    # n by k matrix representing the k iterates.

    function ConvergenceData{T}(n::Int, max_iter::Int) where T<:AbstractFloat

        new(n, max_iter, ones(max_iter), ones(max_iter), ones(n, max_iter))

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
function A_norm!(convergence_data::ConvergenceData, A, X::AbstractMatrix) 

    M = A * X

    convergence_data.relative_error_norm = [

        sqrt( 

            dot(@view(X[:, k]), @view(M[:, k]))

        ) for k in 1:convergence_data.iter_number
    ]

end

function compute_error_norm!(
    convergence_data::ConvergenceData,
    exact_solution  ::Vector,
    A)

    # This should be an (n x k) error matrix.
    error_matrix = exact_solution .- convergence_data.iterates

    # Update error.
    A_norm!(convergence_data, A, error_matrix)

    convergence_data.relative_error_norm .* inv(A_norm(A, exact_solution))

end

function compute_error!(convergence_data::ConvergenceData, exact_solution::AbstractVector, A)

    #A_half        = sqrt.(A)
    errors        = ones(size(A, 1), convergence_data.iter_number)
    solution_norm = A_norm(A, exact_solution)

    for k in 1:convergence_data.iter_number

        errors[:, k] = convergence_data.iterates[:, k] - exact_solution
        tmp          = A * errors[:, k]
        #convergence_data.relative_error_norm[k] = norm(A_half * errors[:, k]) / solution_norm
        convergence_data.relative_error_norm[k] = sqrt(dot(tmp, @view(errors[:, k]))) / solution_norm

    end
end

function compute_residual_norm!(convergence_data::ConvergenceData, A, b)

    b_norm = norm(b)
    xk     = convergence_data.iterates

    for k in 1:convergence_data.iter_number

        convergence_data.relative_residual_norm[k] = norm(b - A * @view(xk[:, k])) * inv(b_norm)

    end
end
