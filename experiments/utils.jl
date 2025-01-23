export FactPrec
export generate_preconditioners, initialize_convergence_data, compute_errors!, 
       get_pcg_variant

const FactPrec{uL, uR, S} = FactorizationPreconditioner{uL, uR, S}

"""
Distinguish between algorithms depending on the preconditioning scheme used.
"""
get_pcg_variant(::Left)  = left_pcg!
get_pcg_variant(::Split) = split_pcg!
get_pcg_variant(::Right) = right_pcg!

"""
Generate preconditioner data structures based on preconditioning scheme, a 
vector of precisions, and the actual preconditioners.
"""
function generate_preconditioners(
    scheme    ::Type{<:PreconditioningScheme},
    Pl        ::AbstractMatrix,
    Pr        ::AbstractMatrix,
    precisions::Vararg{Type{<:AbstractFloat}, N}) where N

    preconditioners = [ FactorizationPreconditioner{u, scheme}(Pl, Pr) for u in precisions]
    

    return preconditioners
end

function generate_preconditioners(
    Pl        ::AbstractMatrix,
    Pr        ::AbstractMatrix,
    precisions::Vararg{Tuple{Type, Type}, N}) where {N}

    preconditioners = [ FactPrec{uL, uR, Split}(Pl, Pr) for (uL, uR) in precisions]
    

    return preconditioners
end

"""
Initialize convergence data.
"""
function initialize_convergence_data(n::Int, max_iter::Int, length::Int)

    return [ ConvergenceData{Float64}(n, max_iter) for _ in 1:length ]

end


function compute_errors!(v_cd, x, A, b, ::AbstractVector, scheme::Union{Type{Left}, Type{Split}}) 

    # Compute error in the A-norm for all runs.
    for k in eachindex(v_cd)
        compute_error!(    v_cd[k], x, A)
        compute_backward_error!(v_cd[k], A, b)
    end

end

function compute_errors!(v_cd, x, A, b, v_prec::AbstractVector, scheme::Type{Right}) 

    MLinv = inv(v_prec[1].Pl)
    MRinv = inv(v_prec[1].Pr)
    Aprec = (A * MRinv) * MLinv 
    #Aprecleft = (MRinv * MLinv) * A
    #Asplit = MLinv * A * MRinv
    #println("Condition number of unpreconditioned matrix: ", cond(Matrix(A)))
    #println("Condition number of left preconditioned matrix: ", cond(Matrix(Aprecleft)))
    #kappa = cond(Matrix(Aprec))
    #println("Condition number of right preconditioned matrix: ", kappa)
    #println("Condition number of split preconditioned matrix: ", )
    y = Aprec\b

    # Compute error in the A-norm for all runs.
    for k in eachindex(v_cd)
        compute_error!(    v_cd[k], y, A, v_prec[k])
        compute_backward_error!(v_cd[k], A, v_prec[k], b)
    end

end
