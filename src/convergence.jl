export ConvergenceData
export compute_error_norm!, A_norm, A_norm!

mutable struct ConvergenceData{T}

    n                     ::Int
    iter_number           ::Int
    iterations            ::Vector{Int}
    relative_residual_norm::Vector{T}
    relative_error_norm   ::Vector{T}
    iterates              ::Matrix{T}

    function ConvergenceData{T}(n::Int, max_iter::Int) where T<:AbstractFloat

        new(
        n, 
        max_iter, 
        collect(1:max_iter),
        ones(max_iter),
        ones(max_iter),
        ones(n, max_iter))

    end

end

A_norm(A, x::AbstractVector) = sqrt(dot(x, A*x))

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

    #convergence_data.relative_error_norm .* A_norm(A, exact_solution)

end

function compute_error!(convergence_data::ConvergenceData, exact_solution::AbstractVector, A)

    A_half        = sqrt.(A)
    errors        = ones(size(A, 1), convergence_data.iter_number)
    solution_norm = norm(exact_solution)

    for k in 1:convergence_data.iter_number

        errors[:, k] = convergence_data.iterates[:, k] - exact_solution
        convergence_data.relative_error_norm[k] = norm(A_half * errors[:, k]) / solution_norm

    end
end
