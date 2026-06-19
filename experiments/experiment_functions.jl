using MixedPrecisionPCG
using BFloat16s, JSON, LinearAlgebra

export low_precision_preconditioner, perturb_preconditioner!, runpcgexperiments, upper_bound, write_heatmap_data, write_to_file 

export Experiment, Error, Residual, AbstractMetric 

abstract type AbstractMetric end

struct Error    <: AbstractMetric end
struct Residual <: AbstractMetric end

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

mutable struct Experiment{T <: PreconditioningScheme} 
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

  function Experiment{T}(experiment::Experiment) where T <: PreconditioningScheme

    new(experiment.ls, experiment.preconditioner, experiment.max_iter, experiment.precisions)

  end

end

get_unique_precisions(experiment::Experiment{Left}) = experiment.precisions

get_unique_precisions(experiment::Experiment{T}) where T <: AbstractSplit = experiment.precisions

getproblemsize(experiment::Experiment) = size(experiment.ls.A, 1)

get_scheme(::Experiment{T}) where T = T

function perturb_preconditioner!(experiment::Experiment, perturbation::AbstractMatrix) 

  experiment.preconditioner += perturbation

end

function runpcgexperiments(experiment::Experiment{T}) where T <: PreconditioningScheme

		scheme = get_scheme(experiment)

    # Initialize AccuracyDataSeries data structure.
		ads = AccuracyDataSeries{Float64}(experiment.precisions, scheme, experiment.max_iter) 

		println("Preconditioning scheme: " * string(scheme))

		n = getproblemsize(experiment)

    # Iterate over different precision choices.

    for precision in experiment.precisions

        # Extract precisions
        uL, uR = getprecisions(scheme, precision)

        ML, MR = experiment.preconditioner, experiment.preconditioner'

        # Generate preconditioner data structure with corresponding precisions.
        M = FactorizationPreconditioner{uL, uR, scheme}(ML, MR)

        # Initialize convergence data structure.
        cd = ConvergenceData{Float64}(n, experiment.max_iter)

        # Run PCG on linear system.
        pcg!(cd, experiment.ls.A, M, experiment.ls.b, experiment.ls.x0, experiment.max_iter)

        println("  Solved system using precisions " * string(precision))

        # Compute accuracy data (norms of true/updated residuals, errors, etc.)
				compute_accuracy_data!(scheme, ads[precision, scheme], cd, experiment.ls, experiment.preconditioner)

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



# Tell JSON to only serialize dictionary for the AccuracyData type.
JSON.lower(ad::AccuracyData)        = ad.metric_dictionary
JSON.lower(ads::AccuracyDataSeries) = ads.series

function write_to_file(
		filename  ::String,
		ads       ::AccuracyDataSeries,
		experiment::Experiment
		)

		upper_bounds = Dict(
				"ErrorNorm"   => upper_bound(experiment, Error),
				"TrueResNorm" => upper_bound(experiment, Residual),
		)

		if !isdir("json_data")

				mkdir(pwd() * "/json_data")

		end

		tmp              = keys(ads.series) 
		precision_vector = sort([key for key in tmp], rev = true)

    data = Dict(
              "metric_data" => ads,
              "bounds"      => upper_bounds,
              "max_iter"    => experiment.max_iter,
              "precision_labels" => precision_vector
          )

	
		JSON.json( pwd() * "/json_data/" * filename, data; pretty = true, allownan = true )

end

"""
Compute minimum and corresponding index of each given metric.
"""
function Base.findmin(ad::AccuracyData, metric::String) 

		minimum = 0.0
		idx     = 0

		metric_data  = ad.metric_dictionary[metric]
		minimum, idx = findmin(metric_data[ map(x -> x > 0.0, metric_data)])

		return idx, minimum
end

function multiplier(experiment::Experiment, metric::String) 

		# We need to transform data to backward or forward error, respectively. 
		normAx               = experiment.ls.normA * experiment.ls.normx
		residual_multiplier  = normAx / ( norm(experiment.ls.b) + normAx )

		multiplier = 1.0

		if metric == "ErrorNorm"

				multiplier = normAx

		elseif metric == "TrueResNorm"

				multiplier = residual_multiplier

		end

		return multiplier

end


"""
Generate forward and backward error as well as iteration count matrices for split preconditioned PCG runs.  
"""
function transform_data_to_heatmap(experiment::Experiment, ads::AccuracyDataSeries, metrics...)

		tmp = keys(ads.series) 

		precision_vector = sort([key for key in tmp], rev = true)

		n_runs           = length(ads)         # Total number of combinations of different precisions
		n_precisions     = Int( sqrt(n_runs) ) # Number of different precisions 

		minima_matrices  = Dict(metric => zeros(n_precisions, n_precisions) for metric in metrics)

		iteration_count_matrix = Matrix{Int}(undef, (n_precisions, n_precisions))

    for metric in metrics

      metric_array         = zeros(n_runs)
      iterationcount_array = zeros(n_runs)

      for (i, precision) in enumerate(precision_vector)

        iterationcount_array[i], metric_array[i] = findmin(ads[precision], metric)

      end

      metric_multiplier       = multiplier(experiment, metric)
      minima_matrices[metric] = metric_multiplier .* reshape(metric_array, n_precisions, n_precisions)
      iteration_count_matrix  = reshape(iterationcount_array, n_precisions, n_precisions)

    end

		return minima_matrices, iteration_count_matrix

end


function write_heatmap_data(filename::String, experiment::Experiment, ads::AccuracyDataSeries) 

	minima_matrices, iteration_count_matrix = transform_data_to_heatmap(experiment, ads, "ErrorNorm", "TrueResNorm")

  precisions = experiment.precisions
  tmp        = [precisiontolabel(precisions[j]) for precisions in precisions for j in eachindex(precisions)]

  precision_vector = sort(unique(tmp), rev = true)
  n                = length(precision_vector) 

	data = Dict(
			"FE_matrix"        => minima_matrices["ErrorNorm"],
			"BE_matrix"        => minima_matrices["TrueResNorm"],
			"ic_matrix"        => iteration_count_matrix,
			"precision_labels" => precision_vector,
      "precision_matrix" => reshape(sort([process_label(precision, Split) for precision in experiment.precisions], rev = true), n, n)
	)

	if !isdir("json_data")

		mkdir(pwd() * "/json_data")

	end

	JSON.json( pwd() * "/json_data/" * filename, data; pretty = true)
end
