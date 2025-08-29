using LinearAlgebra

export AccuracyData, AccuracyDataSeries
export compute_accuracy_data!, get_iternumber

"""
    AccuracyData{T}(iter_number, trueresnorm, updatedresnorm, errornorm,
        resgapnorm, max_ratios)

    Structure holding accuracy measures of different PCG runs as iterations go by.

    
# Arguments
- `iter_number::Int`: Number of iterations
- `trueresnorm::Vector{Vector{T}}`: Vector true residual norms of different PCG runs.
- `updatedresnorm::Vector{Vector{T}}`
- `errornorm::Vector{Vector{T}}`
- `resgapnorm::Vector{Vector{T}}`
- `max_ratios::Dict{Int, Tuple{Float64, Int}}`
   
    
"""
mutable struct AccuracyData{T}

    iter_number   ::Int
    trueresnorm   ::Vector{T}
    updatedresnorm::Vector{T}
    errornorm     ::Vector{T}
    resgapnorm    ::Vector{T}
    #max_ratios    ::Dict{Int, Tuple{Float64, Int}}

    function AccuracyData{T}(iter_number::Int) where {T}

        trueresnorm    = zeros(iter_number) 
        updatedresnorm = zeros(iter_number) 
        errornorm      = zeros(iter_number) 
        resgapnorm     = zeros(iter_number) 
        #max_ratios     = Dict(i => (0.0, 0)  for i in 1:n_precisions)

        #new(iter_number, trueresnorm, updatedresnorm, errornorm, resgapnorm, max_ratios)
        new(iter_number, trueresnorm, updatedresnorm, errornorm, resgapnorm)

    end

end

mutable struct AccuracyDataSeries{T}

    series::Vector{AccuracyData{T}}

    function AccuracyDataSeries{T}(collection_size::Integer, iter_number::Integer) where T

        series = [AccuracyData{T}(iter_number) for _ in 1:collection_size]

        new(series)

    end

end

get_iternumber(ads::AccuracyDataSeries) = ads.series[1].iter_number

Base.length(ads::AccuracyDataSeries)               = length(ads.series)

Base.getindex(ads::AccuracyDataSeries, i::Integer) = ads.series[i]

Base.eachindex(ads::AccuracyDataSeries)            = eachindex(ads.series)

printlnindent(s::AbstractString) = println("  " * s)


function compute_accuracy_data!(
    scheme::Type{<:PreconditioningScheme},
    ad            ::AccuracyData{T},
    cd            ::ConvergenceData{T},
    ls            ::LinearSystem{T},
    preconditioner::AbstractMatrix) where T

    ad.trueresnorm    =    true_residual_norm(cd, ls)

    printlnindent("Computed true residual")

    ad.updatedresnorm = updated_residual_norm(cd, ls)

    printlnindent("Computed updated residual")

    ad.errornorm      =           error_Anorm(cd, ls)

    printlnindent("Computed A-norm of the error")

    ad.resgapnorm     = residualgapnorm(cd, ls, scheme, preconditioner)

    printlnindent("Computed norm of the residual gap\n")

    #ad.max_ratios     =   max_iterate_ratio(cd, ls.x)

    #println("Computed maximum ratios")
    
end

function Base.show(io::IO, ad::AccuracyData{T}) where T<:AbstractFloat

    println(io, "Accuracy data: ")
    println(io, " - Relative true residual norm:    ", typeof(ad.trueresnorm))
    println(io, " - Relative updated residual norm: ", typeof(ad.updatedresnorm))
    println(io, " - Relative in the A-norm:         ", typeof(ad.errornorm))
    #println(io, " - Maximum iterate ratios:         ", typeof(ad.max_ratios))

    println(io, "\nComputations ran for ",  ad.iter_number, " iterations.")
    println(io, "Achieved relative residual norm: ", ad.trueresnorm[end]
    )
end

"""
Compute updated residual norm ||rk|| / ||A|| ||x||
"""

function updated_residual_norm(cd::ConvergenceData, ls::LinearSystem)
    
    norm_rk = [ norm( @view(cd.updated_residuals[:, k]) ) for k in 1:cd.iter_number ]

    return inv(ls.normA * ls.normx) .* norm_rk
    
end


"""
Compute relative (true) residual norm ||b - A xk|| / ||A|| ||x||.
"""
function true_residual_norm(cd::ConvergenceData, ls::LinearSystem)

    trueresnorm = zeros(cd.iter_number)

    xk = cd.iterates

    for k in 1:cd.iter_number

        trueresnorm[k] = norm(ls.b - ls.A * @view xk[:, k]) 

    end

    return inv(ls.normA * ls.normx) .* trueresnorm

end

"""
Compute relative error in the A-norm: ||x - xk||_A / ||x - x0||_A.
"""
function error_Anorm(cd::ConvergenceData, ls::LinearSystem)

    errornorm = zeros(cd.iter_number)

    xk            = cd.iterates
    denominator   = sqrt(norm(ls.A)) * norm(ls.x)

    for k in 1:cd.iter_number

        errornorm[k] = A_norm(ls.A, ls.x - @view xk[:, k])

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

    resgapnorm = zeros(cd.iter_number)

    xk = cd.iterates
    rk = process_residuals(copy(cd.updated_residuals), scheme(), preconditioner)

    for k in 1:cd.iter_number

        resgapnorm[k] = norm(ls.b - ls.A * @view(xk[:, k]) -  @view(rk[:, k]))

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
