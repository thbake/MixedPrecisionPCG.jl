export FactPrec
export getprecisions

const FactPrec{uL, uR, S} = FactorizationPreconditioner{uL, uR, S}

function getprecisions(::Type{Left}, precisions::Type{<:AbstractFloat})

    return precisions, precisions

end

function NumericalExperiments.getprecisions(::Type{<:AbstractSplit}, precisions::Tuple{Type, Type})

    return precisions[1], precisions[2]
end


