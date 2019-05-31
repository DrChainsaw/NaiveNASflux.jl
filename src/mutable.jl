
abstract type AbstractMutableComp end

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


mutable struct MutableLayer <: AbstractMutableComp
    layer
end
(m::MutableLayer)(x) = layer(m)(x)
layer(m::MutableLayer) = m.layer
layertype(m::MutableLayer) = layertype(layer(m))

NaiveNASlib.nin(m::MutableLayer) = nin(layer(m))
NaiveNASlib.nout(m::MutableLayer) = nout(layer(m))

NaiveNASlib.mutate_inputs(m::MutableLayer, inputs::AbstractArray{<:Integer,1}...) =
mutate_inputs(layertype(m), m, inputs...)
function NaiveNASlib.mutate_inputs(::ParLayer, m::MutableLayer, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    l = layer(m)
    w = select(inputs[1], weights(l), indim(l))
    b = copy(bias(l))
    newlayer(m, w, b)
end
NaiveNASlib.mutate_outputs(m::MutableLayer, outputs::AbstractArray{<:Integer,1}) =
mutate_outputs(layertype(m), m, outputs)
function NaiveNASlib.mutate_outputs(::ParLayer, m::MutableLayer, outputs::AbstractArray{<:Integer,1})
    l = layer(m)
    w = select(outputs, weights(l), outdim(l))
    b = select(outputs, bias(l), 1)
    newlayer(m, w, b)
end

newlayer(m::MutableLayer, w, b) = newlayer(layertype(m), m, w, b)

newlayer(::ParDense, m::MutableLayer, w, b) = m.layer = Dense(param(w), param(b), deepcopy(layer(m).σ))
newlayer(::ParConv, m::MutableLayer, w, b) = m.layer = setproperties(layer(m), (weight=param(w), bias=param(b), σ=deepcopy(layer(m).σ)))


mutable struct LazyMutable <: AbstractMutableComp
    mutable
    inputs::AbstractArray{<:Integer, 1}
    outputs::AbstractArray{<:Integer, 1}
end
LazyMutable(m::AbstractMutableComp) = LazyMutable(m, 1:nin(m), 1:nout(m))
LazyMutable(m, nin::Integer, nout::Integer) = LazyMutable(m, 1:nin, 1:nout)

(m::LazyMutable)(x) = dispatch(m, m.mutable, x)
dispatch(m::LazyMutable, mutable::AbstractMutableComp, x) = mutable(x)

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
