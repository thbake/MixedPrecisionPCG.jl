using MixedPrecisionPCG
using Random, JSON3, BFloat16s, JSON

export geomdist_eigvalmatrix, low_precision_preconditioner, cond_experiment, upper_bound, 
	   runpcgexperiments, strakos_mat, mat_prec, write_to_file, write_heatmap_data

export Experiment, Error, Residual, AbstractMetric, MinimaData, IndexData

Random.seed!(1234)

abstract type AbstractMetric end

struct Error    <: AbstractMetric end
struct Residual <: AbstractMetric end

abstract type AbstractHeatmapData end

struct IndexData  <: AbstractHeatmapData end
struct MinimaData <: AbstractHeatmapData end



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

struct Experiment{T <: PreconditioningScheme} 
	ls            ::LinearSystem
	preconditioner::AbstractMatrix
	max_iter      ::Integer
	precisions    ::AbstractVector

	function Experiment{T}(
			n         ::Integer,
			lambda_min::AbstractFloat,
			lambda_max::AbstractFloat,
			rho       ::AbstractFloat,
			i         ::Integer,
			max_iter  ::Integer,
			precisions::AbstractVector) where T <: PreconditioningScheme

		A, L = mat_prec(n, lambda_min, lambda_max, rho, i) # Generate matrix and preconditioner.
		b    = inv(sqrt(n)) .* ones(n)                     # Generate right-hand side.
		x    = A \ b                                       # Solve system directly for reference.
		ls   = LinearSystem(A, b, x, zeros(n))             # Construct linear system.

		new(ls, L, max_iter, precisions)
	end
end

get_scheme(::Experiment{T}) where T = T

function runpcgexperiments(experiment::Experiment{T}) where T <: PreconditioningScheme

    # Initialize AccuracyDataSeries data structure.
    ads = AccuracyDataSeries{Float64}(length(experiment.precisions), experiment.max_iter) 

    n  = size(experiment.ls.A, 1)

	println("Preconditioning scheme: " * string(get_scheme(experiment)))

	scheme = get_scheme(experiment)

    # Iterate over different precision choices.
    for i in eachindex(experiment.precisions)

        # Extract precisions
        uL, uR = getprecisions(scheme, experiment.precisions[i])

        ML, MR = experiment.preconditioner, experiment.preconditioner'

        # Generate preconditioner data structure with corresponding precisions.
        M = FactorizationPreconditioner{uL, uR, scheme}(ML, MR)

        # Initialize convergence data structure.
        cd = ConvergenceData{Float64}(n, experiment.max_iter)

        # Run PCG on linear system.
        pcg!(cd, experiment.ls.A, M, experiment.ls.b, experiment.ls.x0, experiment.max_iter)

        println("  Solved system " * string(i))

        # Compute accuracy data (norms of true/updated residuals, errors, etc.)
        compute_accuracy_data!(scheme, ads[i], cd, experiment.ls, experiment.preconditioner)

    end

    return ads
end

function upper_bound(
	experiment::Experiment,
	bound_type::Type{<:AbstractMetric})

    u = 0.5 * eps(Float64)

	M = Symmetric(experiment.preconditioner * experiment.preconditioner')
	A = experiment.ls.A
	kappaM = cond(M)
	kappaA = (bound_type == Error) ? cond(A) : 1.0

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
		filename  ::String,
		ads       ::AccuracyDataSeries,
		bound     ::AbstractFloat,
		experiment::Experiment,
		metric    ::Type{<:AbstractMetric})

	label_dict =  Dict(

        Float64  => "fp64",
        Float32  => "fp32",
        Float16  => "fp16",
        BFloat16 => "b16"
    )

	processed_labels = [process_label(k, label_dict, get_scheme(experiment)) for k in experiment.precisions]

	value_array   = zeros(experiment.max_iter)
	metric_data   = Dict( label => copy(value_array) for label in processed_labels )
	metric_symbol = get_metric_symbol(metric) # :errornorm or :trueresnorm

	for (i, precision_label) in enumerate(processed_labels)

		metric_data[precision_label] = getproperty(ads[i], metric_symbol)

	end

	metric_data["bound"] = repeat([bound], experiment.max_iter)

	if !isdir("json_data")

		mkdir(pwd() * "/json_data")

	end

	JSON3.write(pwd() * "/json_data/" * filename, metric_data, allow_inf = true)

end


function transform_data_to_heatmap(experiment::Experiment, ads::AccuracyDataSeries)

	metric_symbols = [:errornorm, :trueresnorm]

	# Isolate data first in a vector. Dictionary with each value being a vector of vectors.
	tmp_data = Dict(metric => [ getproperty(ads[i], metric) for i in eachindex(ads) ] for metric in metric_symbols )

	
	# Dictionary, where each value is a pair of floats and integers.
	min_idx_pairs = Dict( 

		metric => Vector{Tuple{Float64, Int}}(undef, experiment.max_iter) for metric in metric_symbols 

	) 

	# Apply Boolean mask and find corresponding minima and indices.
	for metric in metric_symbols

		tmp_data_metric = tmp_data[metric]

		for (i, vector) in enumerate(tmp_data_metric)

			masked_data_array = vector[ map(x -> x > 0.0, vector)]

			min_idx_pairs[metric][i] = findmin(masked_data_array)

		end

	end

	# We need to transform data to backward or forward error, respectively. 
	normAx               = experiment.ls.normA * experiment.ls.normx
	residual_multiplier  = normAx / ( norm(experiment.ls.b) * normAx )

	multiplier = Dict( 

				  metric 

				  => metric == :trueresnorm ? residual_multiplier

				  : sqrt(experiment.ls.normA) for metric in metric_symbols 
				)

	n_precisions    = 4 # Number of different precisions

	minima_matrices = Dict(metric => zeros(n_precisions, n_precisions) for metric in metric_symbols)

	iteration_count_matrix = Matrix{Int}(undef, (n_precisions, n_precisions))

	for j in 1:n_precisions, i in 1:n_precisions

		linear_index = i + (j - 1) * n_precisions

		for metric in metric_symbols

			min_metric_value             = min_idx_pairs[metric][linear_index][1]
			minima_matrices[metric][j,i] = min_metric_value * multiplier[metric]

		end

		iteration_count_matrix[j,i] = min_idx_pairs[:errornorm][linear_index][2]

	end

	return minima_matrices, iteration_count_matrix

end

get_label_array(experiment::Experiment{Left}) = experiment.precisions

function get_label_array(experiment::Experiment{Split}) 

	n_precisions = length(experiment.precisions)

	step = Int( sqrt(n_precisions) )

	return [ experiment.precisions[i][1] for i in 1:step:n_precisions ]
end

function write_heatmap_data(filename::String, experiment::Experiment{T}, ads::AccuracyDataSeries) where T <: PreconditioningScheme

	label_dict =  Dict(

        Float64  => "fp64",
        Float32  => "fp32",
        Float16  => "fp16",
        BFloat16 => "b16"
    )

	minima_matrices, iteration_count_matrix = transform_data_to_heatmap(experiment, ads)

	label_array = get_label_array(experiment)

	println(typeof(minima_matrices[:errornorm]))

	data = Dict(
			"FE_matrix"        => minima_matrices[:errornorm],
			"BE_matrix"        => minima_matrices[:trueresnorm],
			"ic_matrix"        => iteration_count_matrix,
			"precision_labels" => [ label_dict[label] for label in label_array ]
			)

	if !isdir("json_data")

		mkdir(pwd() * "/json_data")

	end

	#JSON3.write( pwd() * "/json_data/" * filename, data )
	JSON.json( pwd() * "/json_data/" * filename, data )

end
