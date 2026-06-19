using LinearAlgebra, BFloat16s

export AccuracyData, AccuracyDataSeries
export compute_accuracy_data!, precisiontolabel, process_label

process_label(precision::DataType, ::Type{Left}) = precisiontolabel(precision)

process_label(
  precisions::Tuple{T1, T2},
            ::Type{<:AbstractSplit}
 ) where {T1 <: DataType, T2 <: DataType} = "(" * 
											precisiontolabel(precisions[1]) *
											", " * 
											precisiontolabel(precisions[2]) *
											")"

precisiontolabel(float_type::Type{Float64}) = "fp64"
precisiontolabel(float_type::Type{Float32}) = "fp32"
precisiontolabel(float_type::Type{Float16}) = "fp16"
precisiontolabel(float_type::Type{BFloat16}) = "bfloat16"

"""
		AccuracyData{T}(max_iter, metric_dictionary)

    Structure holding accuracy measures of different PCG runs as iterations go by.

    
# Arguments
- `max_iter::Int`: Number of iterations
- `metric_dictionary::Int`: Dictionary holding accuracy metrics as keys and the corresponding values across iterations.
   
    
"""
mutable struct AccuracyData{T}

    max_iter         ::Int
    metric_dictionary::Dict{String, Vector{T}}

    function AccuracyData{T}(max_iter::Int) where {T}

      metric_dictionary = Dict{String, Vector{T}}(
        "TrueResNorm"    => zeros(max_iter),
        "UpdatedResNorm" => zeros(max_iter),
        "ErrorNorm"      => zeros(max_iter),
        "ResGapNorm"     => zeros(max_iter)
      )

      new(max_iter, metric_dictionary)

    end

end

Base.getindex(ad::AccuracyData, key::String) = ad.metric_dictionary[key]

"""
		AccuracyDataSeries{T}(series)

    Structure holding AccuracyData structures corresponding to the precisions with which the preconditioners were applied.
    For instance: AccuracyDataSeries{T, U}[(Float32, Float64)] => AccuracyData{T} 
    
# Arguments
- `max_iter::Int`: Number of iterations
- `metric_dictionary::Int`: Dictionary holding accuracy metrics as keys and the corresponding values across iterations.
   
    
"""
mutable struct AccuracyDataSeries{T} 

		series::Dict{String, AccuracyData{T}}

    function AccuracyDataSeries{T}(
                precisions::Vector{U},
                scheme    ::Type{<:PreconditioningScheme},
                max_iter  ::Integer
             ) where {T, U}

        series = Dict( process_label(precision, scheme) => AccuracyData{T}(max_iter) for precision in precisions )

        new(series)

    end

end

Base.length(ads::AccuracyDataSeries) = length(ads.series)

Base.getindex(ads::AccuracyDataSeries, key::U, scheme::Type{T}) where {T <: PreconditioningScheme, U} = ads.series[process_label(key, scheme)] 

Base.getindex(ads::AccuracyDataSeries{T}, key::String) where {T} = ads.series[key] 

Base.eachindex(ads::AccuracyDataSeries) = eachindex(ads.series)

printlnindent(s::AbstractString) = println("  " * s)


function compute_accuracy_data!(
    scheme::Type{<:PreconditioningScheme},
    ad            ::AccuracyData{T},
    cd            ::ConvergenceData{T},
    ls            ::LinearSystem{T},
    preconditioner::AbstractMatrix) where T

		ad.metric_dictionary["TrueResNorm"]    = true_residual_norm(cd, ls)

    printlnindent("Computed true residual")

    ad.metric_dictionary["UpdatedResNorm"] = updated_residual_norm(cd, ls)

    printlnindent("Computed updated residual")

		ad.metric_dictionary["ErrorNorm"]      = error_Anorm(cd, ls)

    printlnindent("Computed A-norm of the error")

		ad.metric_dictionary["ResGapNorm"]     = residualgapnorm(cd, ls, scheme, preconditioner)

    printlnindent("Computed norm of the residual gap\n")

end

function Base.show(io::IO, ad::AccuracyData{T}) where T<:AbstractFloat

    println(io, "Accuracy data: ")
		println(io, " - Relative true residual norm:    ", typeof(ad.metric_dictionary["TrueResNorm"]))
		println(io, " - Relative updated residual norm: ", typeof(ad.metric_dictionary["UpdatedResNorm"]))
		println(io, " - Relative in the A-norm:         ", typeof(ad.metric_dictionary["ErrorNorm"]))

    println(io, "\nComputations ran for ",  ad.max_iter, " iterations.")
		println(io, "Achieved relative residual norm: ", ad.metric_dictionary["TrueResNorm"][end]
    )
end

"""
Compute updated residual norm ||rk|| / ||A|| ||x||
"""

function updated_residual_norm(cd::ConvergenceData, ls::LinearSystem)
    
	norm_rk = zeros(cd.max_iter)

	for k in 1:cd.max_iter

		tmp = norm(@view cd.updated_residuals[:, k])

    norm_rk[k] = ( isnan(tmp) || isinf(tmp) ) ? 0.0 : tmp
	end

    return inv(ls.normA * ls.normx) .* norm_rk
    
end


"""
Compute relative (true) residual norm ||b - A xk|| / ||A|| ||x||.
"""
function true_residual_norm(cd::ConvergenceData, ls::LinearSystem)

    trueresnorm = zeros(cd.max_iter)

    xk = cd.iterates

    for k in 1:cd.max_iter

		tmp = norm(ls.b - ls.A * @view xk[:, k]) 

		trueresnorm[k] = isnan(tmp) ? 0.0 : tmp

    end

    return inv(ls.normA * ls.normx) .* trueresnorm

end

"""
Compute relative error in the A-norm: ||x - xk||_A / ||x - x0||_A.
"""
function error_Anorm(cd::ConvergenceData, ls::LinearSystem)

    errornorm = zeros(cd.max_iter)

    xk            = cd.iterates
    denominator   = sqrt(norm(ls.A)) * norm(ls.x)

    for k in 1:cd.max_iter

		tmp = A_norm(ls.A, ls.x - @view xk[:, k])

		errornorm[k] = isnan(tmp) ? 0.0 : tmp
    end

    return inv(denominator) .* errornorm 


end

"""
Process residuals rk for computing the residual gap. In the left preconditioned
case there is nothing to do since the left preconditioner does not affect
the bound on the residual gap.
"""
process_residuals(rk::AbstractMatrix, ::Left, ::AbstractMatrix)    = rk


"""
Process residuals rk for computing the residual gap. In the split preconditioned
case we need to unprecondition the residuals at each iteration.
"""
process_residuals(rk::AbstractMatrix, ::Split, ML::AbstractMatrix) = rk

process_residuals(rk::AbstractMatrix, ::SaadSplit, ML::AbstractMatrix) = ML * rk


"""
Compute the relative residual gap norm 
 
    ||b - A xk - rk|| / ||A|| ||x|| 

for the left preconditioned case or 

    ||b - A xk - ML rk|| / ||A|| ||x|| 

for the split preconditioned case.
"""

function residualgapnorm(
    cd            ::ConvergenceData,
    ls            ::LinearSystem,
    scheme        ::Type{<:PreconditioningScheme},
    preconditioner::AbstractMatrix)

    resgapnorm = zeros(cd.max_iter)

    xk = cd.iterates
    rk = process_residuals(copy(cd.updated_residuals), scheme(), preconditioner)

    for k in 1:cd.max_iter

		tmp = norm(ls.b - ls.A * @view(xk[:, k]) -  @view(rk[:, k]))

		resgapnorm[k] = isnan(tmp) ? 0.0 : tmp

    end

    return inv(ls.normA * ls.normx) .* resgapnorm

end



"""
Given a symmetric/Hermitian positive definite (SPD/HPD) matrix A and a vector x 
compute the error in the A-norm.
"""
A_norm(A, x::AbstractVector) = sqrt(dot(x, A*x))

"""
Compute ratio that maximizes the norm of iterates.
"""
function max_iterate_ratio(cd::ConvergenceData, x::Vector{T}) where T<:AbstractFloat

    val, idx = findmax([norm(iterate) for iterate in eachcol(cd.iterates)])

    max_ratio = val / norm(x)

    return max_ratio, idx

end
