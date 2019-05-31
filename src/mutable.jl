
abstract type AbstractMutableComp{T} end

#Generic helper functions

function select(elements, pars::AbstractArray{T,N}, dim) where {T, N}
    indskeep = filter(ind -> ind > 0, elements)
    newmap = elements .> 0

    psize = collect(size(pars))
    psize[dim] = length(newmap)
    newpars = zeros(T, psize...)

    assign = repeat(Any[Colon()], N)
    assign[dim] = newmap

    # Indices to
    access = repeat(Any[Colon()], N)
    access[dim] = indskeep

    newpars[assign...] = pars[access...]
    return newpars
end

#Generic helper functions end


mutable struct MutableLayer{T} <: AbstractMutableComp{T}
    layer::T
end
(m::MutableLayer)(x) = m.layer(x)

NaiveNASlib.nin(m::MutableLayer) = nin(m.layer)
NaiveNASlib.nout(m::MutableLayer) = nout(m.layer)

function NaiveNASlib.mutate_inputs(m::MutableLayer{<:ParLayer}, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    l = m.layer
    w = select(inputs[1], weights(l), indim(l))
    b = copy(bias(l))
    newlayer(m, w, b)
end

function NaiveNASlib.mutate_outputs(m::MutableLayer{<:ParLayer}, outputs::AbstractArray{<:Integer,1})
    l = m.layer
    w = select(outputs, weights(l), outdim(l))
    b = select(outputs, bias(l), 1)
    newlayer(m, w, b)
end

newlayer(m::MutableLayer{<:Dense}, w, b) = m.layer = Dense(param(w), param(b), deepcopy(m.layer.σ))
newlayer(m::MutableLayer{<:ParConv}, w, b) = m.layer = setproperties(m.layer, (weight=param(w), bias=param(b), σ=deepcopy(m.layer.σ)))


mutable struct LazyMutable{T} <: AbstractMutableComp{T}
    mutable::AbstractMutableComp{T}
    inputs::AbstractArray{<:Integer, 1}
    outputs::AbstractArray{<:Integer, 1}
end
LazyMutable(m::AbstractMutableComp{T}) where T = LazyMutable(m, 1:nin(m), 1:nout(m))
LazyMutable(m::AbstractMutableComp{T}, nin::Integer, nout::Integer) where T = LazyMutable(m, 1:nin, 1:nout)

(m::LazyMutable)(x) = dispatch(m, m.mutable, x)
dispatch(m::LazyMutable{T}, mutable::AbstractMutableComp{T}, x) where T = mutable(x)

NaiveNASlib.nin(m::LazyMutable) = length(m.inputs)
NaiveNASlib.nout(m::LazyMutable) = length(m.outputs)

function NaiveNASlib.mutate_inputs(m::LazyMutable, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    m.inputs = inputs[1]
end

function NaiveNASlib.mutate_outputs(m::LazyMutable, outputs::AbstractArray{<:Integer,1})
    m.outputs = outputs
end

function NaiveNASlib.mutate_inputs(m::LazyMutable, nin::Integer...)
    mutate_inputs(m, map(n -> collect(1:n), nin)...)
end

function NaiveNASlib.mutate_outputs(m::LazyMutable, nout::Integer)
    mutate_outputs(m, 1:nout)
end
