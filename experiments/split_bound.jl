using MixedPrecisionPCG, .NumericalExperiments
using LinearAlgebra, Random

Random.seed!(123)

n = 40

left_preconditioner(kappa, n) = diagm([ kappa^((i - 1) / (n - 1)) for i in 1:n ])

x = rand(n)

function truncate(diagonal::Vector{T}, idx::Int) where T

    tmp        = copy(diagonal)
    eigenvalue = diagonal[idx]

    for i in idx:length(diagonal)

        tmp[i] = eigenvalue 

    end

    return diagm(tmp.^2)

end

x0 = zeros(n)

function cond_experiment(
    scheme        ::Type{<:PreconditioningScheme},
    precisions    ::AbstractVector, 
    truncation_idx::Int,
    kappa_range   ::Vector{Float64},
    max_iter      ::Int)

    ad_vector   = Vector{AccuracyData}(undef, length(kappa_range))

    for i in eachindex(kappa_range)

        # Create (left) preconditioner.
        L = left_preconditioner(kappa_range[i], n) 

        # Create A based on a truncation of the eigenvalues of L.
        A = truncate(diag(L), truncation_idx)

        b = A * x

        ls = LinearSystem(A, b, x, x0)

        ad_vector[i] = collect_data(scheme, precisions, ls, L, max_iter)

    end

    return ad_vector
end

d = Float64
s = Float32
h = Float16

splitpcg_precisions = [(s, d)]
kappa_range         = 10.0 .^collect(2:2:10)

ad_vec = cond_experiment(Split, splitpcg_precisions, 25, kappa_range, 150)




