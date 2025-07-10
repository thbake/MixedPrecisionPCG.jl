export FactPrec
export generate_preconditioners, getprecisions, collect_data

const FactPrec{uL, uR, S} = FactorizationPreconditioner{uL, uR, S}


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

function getprecisions(scheme::Type{Left}, precisions::Type{<:AbstractFloat})

    return precisions, precisions

end

function getprecisions(scheme::Type{Split}, precisions::Tuple{Type, Type})

    return precisions[1], precisions[2]
end

function collect_data(
    scheme         ::Type{<:PreconditioningScheme},
    precisions     ::AbstractVector,
    ls             ::LinearSystem,
    preconditioner ::AbstractMatrix,
    max_iter::Int)

    ad = AccuracyData{Float64}(length(precisions), max_iter)

    n  = size(ls.A, 1)

    for i in eachindex(precisions)

        # Extract precisions
        uL, uR = getprecisions(scheme, precisions[i])

        # Generate preconditioner data structure with corresponding precisions.
        M = FactorizationPreconditioner{uL, uR, scheme}(preconditioner, preconditioner')

        # Initialize convergence data structure.
        cd = ConvergenceData{Float64}(n, max_iter)

        println(scheme)

        # Run PCG on linear system.
        pcg!(cd, ls.A, M, ls.b, ls.x0, max_iter, scheme())

        println("Solved system " * string(i))

        # Compute accuracy data (norms of true/updated residuals, errors, etc.)
        compute_accuracy_data!(scheme, ad, cd, ls, preconditioner, i)

    end

    return ad

end

