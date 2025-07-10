using MixedPrecisionPCG
using Random, MATLAB

export runpcgexperiments, geomdist_eigvalmatrix, low_precision_preconditioner,
       cond_experiment,  upper_bound

Random.seed!(1234)

"""
    runpcgexperiments(scheme, v_ls, v_prec, v_iter, precisions)

Run pcg! on a varierty of linear algebraic systems (given in the vector v_ls) 
and compute the corresponding collection of AccuracyData (stored in the vector 
v_ad). 

...
# Arguments
- `scheme    ::Type{  <:PreconditioningScheme}`: type of preconditioning scheme (left/split).
- `v_ls      ::Vector{        LinearSystem{T}}`: vector of linear algebraic systems.
- `v_prec    ::Vector{       <:AbstractMatrix}`: vector of preconditioners (Cholesky factors).
- `v_iter    ::Vector{                    Int}`: vector of maximum number of iterations.
- `precisions::AbstractVector`: vector containing data types corresponding to different precisions.
...
"""
function runpcgexperiments(
    scheme    ::Type{  <:PreconditioningScheme},
    v_ls      ::Vector{        LinearSystem{T}},
    v_prec    ::Vector{       <:AbstractMatrix},
    v_iter    ::Vector{                    Int},
    precisions::AbstractVector) where T <: AbstractFloat

    n_ls = length(v_ls)

    v_ad = Vector{AccuracyData{Float64}}(undef, n_ls)

    for i = 1:n_ls

        v_ad[i] = collect_data(scheme, precisions, v_ls[i], v_prec[i], v_iter[i])

    end

    return v_ad
end


"""
    geomdist_eigvalmatrix(kappa, n)

Given a parameter kappa and a problem size n, computes an SPD matrix A = Q D Q^T
with geometrically distributed eigenvalues from [1, kappa] with spectral 
condition number kappa.
"""
function geomdist_eigvalmatrix(kappa, n)

    # Compute random nxn matrix Y.
    Y = rand(n, n)

    # Compute QR decomposition of Y.
    Q = qr(Y).Q

    # Generate geometrically distributed eigenvalues.
    D = diagm([ kappa^((i - 1) / (n - 1)) for i in 1:n ])

    # Return SPD matrix with prescribed eigenvalues.
    return sparse(Hermitian(Q * D * Q'))
    
end

"""
    low_precision_preconditioner(A; tol)

Given an SPD matrix A and a user prescribed tolerance, computes a low precision
Cholesky factorization as a preconditioner. Two diagonal scaling is employed 
in order to avoid over- or underflow when casting given matrix to half 
precision.
"""
function low_precision_preconditioner(A::AbstractMatrix; tol::AbstractFloat)

    n     = size(A, 1)
    Ah    = two_sided_diagonal_scaling(A, 1.0, tol)
    alpha = 0.0

    while !isposdef(Ah) 

        alpha += 1.0

        Ah = Ah + alpha .* I(n)
        
    end

    Lh = sparse(cholesky(Ah).L)

    return Lh

end

function cond_experiment(
    scheme     ::Type{<:PreconditioningScheme},
    n          ::Int,
    precisions ::AbstractVector, 
    tol        ::AbstractFloat,
    kappa_range::Vector{Float64},
    max_iter   ::Int)

    ad_vector   = Vector{AccuracyData}(undef, length(kappa_range))

    kappa_range_prec = similar(kappa_range)

    x0 = zeros(n)

    for i in eachindex(kappa_range)

        # Create matrix with geometrically distributed eigenvalues.
        A = geomdist_eigvalmatrix(kappa_range[i], n) 

        x = rand(n)

        b = A * x

        ls = LinearSystem(A, b, x, x0)

        # Create low precision Cholesky factor to use as preconditioner.
        Lh = low_precision_preconditioner(A, tol = tol)

        kappa_range_prec[i] = cond(Matrix(Lh))

        ad_vector[i] = collect_data(scheme, precisions, ls, Lh, max_iter)

    end

    return ad_vector, kappa_range_prec
end

function upper_bound(
    kappa    ::Float64,
    pL       ::DataType,
    pA       ::DataType,
    n        ::Int,
    S        ::Int,
    max_ratio::Float64)

    u  = 0.5 * eps(Float64)
    uL = 0.5 * eps(pL)
    uA = 0.5 * eps(pA)

    bound_u  = u  * (max_iter + 1 + (1 + 10max_iter) * max_ratio)
    bound_uA = uA * sqrt(n) * (2max_iter + 1)        * max_ratio
    bound_uL = uL * n^(3/2) * kappa^2 * 2max_iter    * max_ratio

    return bound_u + bound_uA + bound_uL

end
