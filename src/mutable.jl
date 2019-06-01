
abstract type AbstractMutableComp end

#Generic helper functions

function select(pars::AbstractArray{T,N}, elements_per_dim...) where {T, N}
    psize = collect(size(pars))
    assign = repeat(Any[Colon()], N)
    access = repeat(Any[Colon()], N)

    for de in elements_per_dim
        dim = de.first
        elements = de.second

        indskeep = filter(ind -> ind > 0, elements)
        newmap = elements .> 0

        psize[dim] = length(newmap)
        assign[dim] = newmap
        access[dim] = indskeep
    end
    newpars = zeros(T, psize...)
    newpars[assign...] = pars[access...]
    return newpars
end

#Generic helper functions end

"""
    MutableLayer

Wraps a layer in order to allow for mutation as layer structures are typically immutable
"""
mutable struct MutableLayer <: AbstractMutableComp
    layer
end
(m::MutableLayer)(x) = layer(m)(x)
layer(m::MutableLayer) = m.layer
layertype(m::MutableLayer) = layertype(layer(m))

NaiveNASlib.nin(m::MutableLayer) = nin(layer(m))
NaiveNASlib.nout(m::MutableLayer) = nout(layer(m))

function NaiveNASlib.mutate_inputs(m::MutableLayer, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    mutate(layertype(m), m, inputs=inputs[1])
end


NaiveNASlib.mutate_outputs(m::MutableLayer, outputs) = mutate(layertype(m), m, outputs=outputs)

mutate(m::MutableLayer; inputs, outputs) = mutate(layertype(m), m, inputs=inputs, outputs=outputs)
function mutate(::ParLayer, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m))
    l = layer(m)
    w = select(weights(l), outdim(l) => outputs, indim(l) => inputs)
    b = select(bias(l), 1 => outputs)
    newlayer(m, w, b)
end


function mutate(t::ParInvLayer, m::MutableLayer; inputs=missing, outputs=missing)
    @assert any(ismissing.((inputs, outputs))) || inputs == outputs "Try to mutate $inputs and $outputs for invariant layer $(m)!"
    ismissing(inputs) || return mutate(t, m, inputs)
    ismissing(outputs) || return mutate(t, m, outputs)
end

function mutate(::ParDiagonal, m::MutableLayer, inds)
    l = layer(m)
    w = select(weights(l), 1 => inds)
    b = select(bias(l), 1 => inds)
    newlayer(m, w, b)
end

function mutate(::ParLayerNorm, m::MutableLayer, inds)
    proxy = MutableLayer(layer(m).diag)
    mutate(proxy, inputs=inds, outputs=inds)
    m.layer = LayerNorm(layer(proxy))
end

newlayer(m::MutableLayer, w, b) = m.layer = newlayer(layertype(m), m, w, b)

newlayer(::ParDense, m::MutableLayer, w, b) = Dense(param(w), param(b), deepcopy(layer(m).σ))
newlayer(::ParConv, m::MutableLayer, w, b) = setproperties(layer(m), (weight=param(w), bias=param(b), σ=deepcopy(layer(m).σ)))
newlayer(::ParDiagonal, m::MutableLayer, w, b) = Flux.Diagonal(param(w), param(b))


"""
    LazyMutable

Lazy version of MutableLayer in the sense that it does not perform any mutations
until invoked to perform a computation.

This is done by calling the method trigger_mutation on the mutable, meaning that
there must exist a method for the the mutable type in order for something to
happen.

This reduces the need to garbage collect intermediate parameters when both inputs
and outputs are mutated.

Also useable for factory-like designs where the actual layers of a computation graph
are not instantiated until the graph is used.

# Examples
```julia-repl
julia> using NaiveNASflux

julia> struct DenseConfig end

julia> lazy = LazyMutable(DenseConfig(), 2,3)
LazyMutable(DenseConfig(), 1:2, 1:3)

julia> function NaiveNASflux.dispatch(m::LazyMutable, ::DenseConfig, x)
       m.mutable = MutableLayer(Dense(nin(m), nout(m), relu))
       return m.mutable(x)
       end

julia> lazy(Float32[1,2])
Tracked 3-element Array{Float32,1}:
 1.568902f0
 0.556749f0
 2.0417972f0

julia> lazy
LazyMutable(MutableLayer(Dense(2, 3, NNlib.relu)), 1:2, 1:3)
```
"""
mutable struct LazyMutable <: AbstractMutableComp
    mutable
    inputs::AbstractArray{<:Integer, 1}
    outputs::AbstractArray{<:Integer, 1}
end
LazyMutable(m::AbstractMutableComp) = LazyMutable(m, nin(m), nout(m))
LazyMutable(m, nin::Integer, nout::Integer) = LazyMutable(m, 1:nin, 1:nout)

(m::LazyMutable)(x) = dispatch!(m, m.mutable, x)
dispatch!(m::LazyMutable, mutable::AbstractMutableComp, x) = mutable(x)

NaiveNASlib.nin(m::LazyMutable) = length(m.inputs)
NaiveNASlib.nout(m::LazyMutable) = length(m.outputs)

function NaiveNASlib.mutate_inputs(m::LazyMutable, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    m.inputs == inputs[1] && return

    m.mutable = ResetInAndOut(trigger_mutation(m.mutable))
    m.inputs = inputs[1]
end

function NaiveNASlib.mutate_outputs(m::LazyMutable, outputs::AbstractArray{<:Integer,1})
    outputs == m.outputs && return

    m.mutable = ResetInAndOut(trigger_mutation(m.mutable))
    m.outputs = outputs
end

NaiveNASlib.mutate_inputs(m::LazyMutable, nin::Integer...) = mutate_inputs(m, map(n -> collect(1:n), nin)...)


NaiveNASlib.mutate_outputs(m::LazyMutable, nout::Integer) = mutate_outputs(m, 1:nout)

"""
    MutationTriggered

Dispatches mutation for LazyMutable.
"""
struct MutationTriggered
    wrapped
end
# Functionality is opt-in
trigger_mutation(m) = m
trigger_mutation(m::AbstractMutableComp) = MutationTriggered(m)

function dispatch!(lm::LazyMutable, m::MutationTriggered, x)
    mutate(m.wrapped; inputs=lm.inputs, outputs=lm.outputs)
    lm.mutable = m.wrapped
    return lm(x)
end

"""
    ResetInAndOut

Reset input and output when dispatching a LazyMutable
"""
struct ResetInAndOut
    wrapped
end

function dispatch!(lm::LazyMutable, m::ResetInAndOut, x)
    lm.mutable = m.wrapped
    output = lm(x)
    lm.inputs = 1:nin(lm)
    lm.outputs = 1:nout(lm)
    return output
end
