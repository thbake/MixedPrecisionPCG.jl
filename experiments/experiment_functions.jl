using MixedPrecisionPCG
using Random, JSON3, BFloat16s

export geomdist_eigvalmatrix, low_precision_preconditioner, cond_experiment, upper_bound, 
	   runpcgexperiments, strakos_mat, randsvd_spd, mat_prec, write_to_file

export Experiment, Error, Residual, AbstractMetric

Random.seed!(1234)

abstract type AbstractMetric end

struct Error    <: AbstractMetric end
struct Residual <: AbstractMetric end



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

   # Take square root of M to compute Cholesky factors since M is diagonal.
   L = sqrt(M)

   return A, L

end

struct Experiment
	ls            ::LinearSystem
	preconditioner::AbstractMatrix
	max_iter      ::Integer
	precisions    ::AbstractVector

	function Experiment(
			n         ::Integer,
			lambda_min::AbstractFloat,
			lambda_max::AbstractFloat,
			rho       ::AbstractFloat,
			i         ::Integer,
			max_iter  ::Integer,
			precisions::AbstractVector)

		A, L = mat_prec(n, lambda_min, lambda_max, rho, i) # Generate matrix and preconditioner.
		b    = inv(sqrt(n)) .* ones(n)                     # Generate right-hand side.
		x    = A \ b                                       # Solve system directly for reference.
		ls   = LinearSystem(A, b, x, zeros(n))             # Construct linear system.

		new(ls, L, max_iter, precisions)
	end
end

function runpcgexperiments(experiment::Experiment, scheme::Type{<:PreconditioningScheme})

    # Initialize AccuracyDataSeries data structure.
    ads = AccuracyDataSeries{Float64}(length(experiment.precisions), experiment.max_iter) 

	collect_data!(
	  ads,
	  scheme,
	  experiment.precisions,
	  experiment.ls,
	  experiment.preconditioner,
	  experiment.max_iter
	)
    return ads
end

function upper_bound(
	experiment::Experiment,
	bound_type::Type{<:AbstractMetric})

    u = 0.5 * eps(Float64)

	M = Symmetric(experiment.preconditioner * experiment.preconditioner')
	A = experiment.ls.A
	kappaM = cond(M)
	kappaA = bound_type == Error ? cond(A) : 1.0

    return u * sqrt(kappaM) * sqrt(kappaA)

end


# Export data
process_label(precision::DataType, label_dict::Dict, ::Type{Left}) = label_dict[precision]

process_label(
  precisions::Tuple{T1, T2},
  label_dict::Dict,
  ::Type{<:AbstractSplit}
 ) where {T1 <: DataType, T2 <: DataType} = "(" * 
											label_dict[precisions[1]] *
											", " * 
											label_dict[precisions[2]] *
											")"

get_metric_symbol(::Type{Error})    = :errornorm
get_metric_symbol(::Type{Residual}) = :trueresnorm

function write_to_file(
		filename     ::String,
		ads          ::AccuracyDataSeries,
		bound        ::AbstractFloat,
		experiment   ::Experiment,
		scheme       ::Type{<:PreconditioningScheme},
		metric       ::Type{<:AbstractMetric})

	label_dict =  Dict(

        Float64  => "fp64",
        Float32  => "fp32",
        Float16  => "fp16",
        BFloat16 => "b16"
    )

	processed_labels = [process_label(k, label_dict, scheme) for k in experiment.precisions]

	value_array   = zeros(experiment.max_iter)
	metric_data   = Dict( label => copy(value_array) for label in processed_labels )
	metric_symbol = get_metric_symbol(metric)

	for (i, precision_label) in enumerate(processed_labels)

		println(precision_label)
		metric_data[precision_label] = getproperty(ads[i], metric_symbol)


	end

	metric_data["bound"] = repeat([bound], experiment.max_iter)

	if !isdir("json_data")

		mkdir(pwd() * "/json_data")

	end

	JSON3.write(pwd() * "/json_data/" * filename, metric_data, allow_inf = true)

end
