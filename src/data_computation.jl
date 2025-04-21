using LinearAlgebra

export AccuracyData
export compute_accuracy_data!

mutable struct AccuracyData{T}

    iter_number   ::Int
    trueresnorm   ::Vector{Vector{T}}
    updatedresnorm::Vector{Vector{T}}
    errornorm     ::Vector{Vector{T}}
    resgapnorm    ::Vector{Vector{T}}
    max_ratios    ::Dict{Int, Tuple{Float64, Int}}

    function AccuracyData{T}(n_precisions::Int, iter_number::Int) where {T}

        trueresnorm    = [zeros(iter_number) for _ in 1:n_precisions]
        updatedresnorm = [zeros(iter_number) for _ in 1:n_precisions]
        errornorm      = [zeros(iter_number) for _ in 1:n_precisions]
        resgapnorm     = [zeros(iter_number) for _ in 1:n_precisions]
        max_ratios     = Dict(i => (0.0, 0)  for i in 1:n_precisions)

        new(iter_number, trueresnorm, updatedresnorm, errornorm, resgapnorm, max_ratios)


    end

end

function compute_accuracy_data!(
    ad ::AccuracyData{T},
    cd ::ConvergenceData{T},
    ls ::LinearSystem{T},
    idx::Int) where T

    ad.trueresnorm[idx]    = true_residual_norm(cd, ls)
    ad.updatedresnorm[idx] = inv(ls.normA * ls.normx) .* [ norm(cd.updated_residuals[:,k]) for k in 1:cd.iter_number]
    ad.errornorm[idx]      = error_Anorm(cd, ls)
    ad.resgapnorm[idx]     = residualgapnorm(cd, ls)
    ad.max_ratios[idx]     = max_iterate_ratio(cd, ls.x)
    
end

function Base.show(io::IO, ad::AccuracyData{T}) where T<:AbstractFloat

    println(io, "Accuracy data: ")
    println(io, " - Relative true residual norm:    ", typeof(ad.trueresnorm))
    println(io, " - Relative updated residual norm: ", typeof(ad.updatedresnorm))
    println(io, " - Relative in the A-norm:         ", typeof(ad.errornorm))
    println(io, " - Maximum iterate ratios:         ", typeof(ad.max_rations))

    println(io, "\nComputations ran for ",  ad.iter_number, " iterations.")
    println(io, "Achieved relative residual norm: ", ad.trueresnorm[end]
    )
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
    initial_error = A_norm(ls.A, ls.x - ls.x0)

    for k in 1:cd.iter_number

        errornorm[k] = A_norm(ls.A, ls.x - @view xk[:, k])

    end

    return inv(initial_error) .* errornorm 


end

"""
Compute the relative residual gap norm ||b - A xk - rk|| / ||A|| ||x|| for the 
left preconditioned case.
"""

function residualgapnorm(cd::ConvergenceData, ls::LinearSystem)

    resgapnorm = zeros(cd.iter_number)

    xk = cd.iterates
    rk = cd.updated_residuals

    for k in 1:cd.iter_number

        resgapnorm[k] = norm(ls.b - ls.A * @view(xk[:, k]) -  @view(rk[:, k]))

    end

    return inv(ls.normA * ls.normx) .* resgapnorm

end

"""
Compute the relative residual gap norm ||b - A xk - ML rk|| / ||A|| ||x|| for 
the split preconditioned case, where ML is the left preconditioner.
"""

function residualgapnorm(
    cd::ConvergenceData, 
    ls::LinearSystem,
    ML::AbstractMatrix)

    resgapnorm = zeros(cd.iter_number)

    xk = cd.iterates
    rk = ML * cd.updated_residuals # Unprecondition updated residuals.

    for k in 1:cd.iter_number

        resgapnorm[k] = norm(ls.b - ls.A * @view(xk[:, k]) - @view(rk[:, k]))

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
