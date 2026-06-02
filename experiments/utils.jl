export FactPrec
export getprecisions, collect_data!

const FactPrec{uL, uR, S} = FactorizationPreconditioner{uL, uR, S}

function getprecisions(::Type{Left}, precisions::Type{<:AbstractFloat})

    return precisions, precisions

end

function NumericalExperiments.getprecisions(::Type{<:AbstractSplit}, precisions::Tuple{Type, Type})

    return precisions[1], precisions[2]
end


function collect_data!(
    ads            ::AccuracyDataSeries,
    scheme         ::Type{<:PreconditioningScheme},
    precisions     ::AbstractVector,
    ls             ::LinearSystem,
    preconditioner ::AbstractMatrix,
    max_iter::Int)

    n  = size(ls.A, 1)

    println("Preconditioning scheme: " * string(scheme))

    # Iterate over different precision choices.
    for i in eachindex(precisions)

        # Extract precisions
        uL, uR = getprecisions(scheme, precisions[i])

        ML, MR = preconditioner, preconditioner'

        # Generate preconditioner data structure with corresponding precisions.
        M = FactorizationPreconditioner{uL, uR, scheme}(ML, MR)

        # Initialize convergence data structure.
        cd = ConvergenceData{Float64}(n, max_iter)

        # Run PCG on linear system.
        pcg!(cd, ls.A, M, ls.b, ls.x0, max_iter)

        println("  Solved system " * string(i))

        # Compute accuracy data (norms of true/updated residuals, errors, etc.)
        compute_accuracy_data!(scheme, ads[i], cd, ls, preconditioner)

    end

end

