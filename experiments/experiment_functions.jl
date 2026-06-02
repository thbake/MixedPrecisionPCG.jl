using MixedPrecisionPCG
using Random

export runpcgexperiments, geomdist_eigvalmatrix, low_precision_preconditioner,
       cond_experiment, upper_bound, strakos_mat, randsvd_spd, mat_prec

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
	println(n_ls)

    # Initialize vector of AccuracyDataSeries data structures.
    v_ads = [AccuracyDataSeries{Float64}(length(precisions), v_iter[i]) for i in 1:n_ls]

    for i = 1:n_ls

        collect_data!(v_ads[i], scheme, precisions, v_ls[i], v_prec[i], v_iter[i])

    end

    return v_ads
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

strakos_mat(n, λ1, λn, ρ) = diagm(vcat([λ1], [λ1 + (i - 1)/(n - 1) * (λn - λ1) * ρ^(n - i) for i in 2:n-1], [λn]))

function mat_prec(n::Int, l1::Float64, ln::Float64, rho::Float64, cutoff::Int)

   A = strakos_mat(n, l1, ln, rho)

   eigs = diag(A)

   # Truncate first n - "cutoff" eigenvalues <=> Preserve first "cutoff" eigenvalues.
   prec_eigvals = vcat(eigs[1:cutoff], [eigs[cutoff] for _ in 1:n-cutoff])

   M = diagm(prec_eigvals)

   L = cholesky(M).L

   #println(cond((L \ M) / L'))

   return A, L

end

"""
Return smallest iteration step k ∈ N such that the true residual norm meets the 
desired threshold tol.
"""
_smallest_iter(ads::AccuracyDataSeries, tol) = (collect(1:ads[1].iter_number)[ads[1].trueresnorm .<= tol])[1]

function upper_bound(
    ads   ::AccuracyDataSeries,
    kappaM::AbstractFloat,
    kappaA::AbstractFloat,
    n     ::Int,
    tol = 1e-15)

    u = 0.5 * eps(Float64)

    k = _smallest_iter(ads, tol)

    return n * k^2 * u * sqrt(kappaM) * sqrt(kappaA)
    #return n * u * sqrt(kappaM) * sqrt(kappaA)

end
