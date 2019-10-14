
"""
     AbstractMutableComp

Abstract base type for computation units (read layers) which may mutate.
"""
abstract type AbstractMutableComp end
Base.Broadcast.broadcastable(m::AbstractMutableComp) = Ref(m)

# Generic forwarding methods. Just implement layer(t::Type) and wrapped(t::Type) to enable

# Not possible in julia <= 1.1. See #14919
# (m::AbstractMutableComp)(x...) = layer(m)(x...)
layertype(m::AbstractMutableComp) = layertype(layer(m))

NaiveNASlib.nin(m::AbstractMutableComp) = nin(layer(m))
NaiveNASlib.nout(m::AbstractMutableComp) = nout(layer(m))

# Leave some room to override clone
NaiveNASlib.clone(m::AbstractMutableComp;cf=clone) = typeof(m)(map(cf, getfield.(m, fieldnames(typeof(m))))...)

NaiveNASlib.mutate_inputs(m::AbstractMutableComp, inputs::AbstractArray{<:Integer,1}...) = mutate_inputs(wrapped(m), inputs...)
NaiveNASlib.mutate_outputs(m::AbstractMutableComp, outputs) = mutate_outputs(wrapped(m), outputs)

mutate_weights(m::AbstractMutableComp, w) = mutate_weights(wrapped(m), w)

NaiveNASlib.minΔninfactor(m::AbstractMutableComp) = minΔninfactor(layertype(m), layer(m))
NaiveNASlib.minΔnoutfactor(m::AbstractMutableComp) = minΔnoutfactor(layertype(m), layer(m))

NaiveNASlib.compconstraint!(s, m::AbstractMutableComp, data) = NaiveNASlib.compconstraint!(s, layertype(m), m, data)
function NaiveNASlib.compconstraint!(s, l, m::AbstractMutableComp, data) end
NaiveNASlib.compconstraint!(s, l::FluxLayer, m::AbstractMutableComp, data) = NaiveNASlib.compconstraint!(s, l, layer(m), data)


"""
    MutableLayer

Wraps a layer in order to allow for mutation as layer structures are typically immutable
"""
mutable struct MutableLayer <: AbstractMutableComp
    layer
end
(m::MutableLayer)(x...) = layer(m)(x...)
wrapped(m::MutableLayer) = m.layer
layer(m::MutableLayer) = wrapped(m)
layertype(m::MutableLayer) = layertype(layer(m))

function NaiveNASlib.mutate_inputs(m::MutableLayer, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    mutate(layertype(m), m, inputs=inputs[1])
end

NaiveNASlib.mutate_outputs(m::MutableLayer, outputs) = mutate(layertype(m), m, outputs=outputs)

mutate_weights(m::MutableLayer, w) = mutate(layertype(m), m, other=w)

mutate(m::MutableLayer; inputs, outputs, other = l -> ()) = mutate(layertype(m), m, inputs=inputs, outputs=outputs, other=other)

function mutate(::FluxParLayer, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m), other= l -> ())
    l = layer(m)
    otherdims = other(l)
    w = select(weights(l), outdim(l) => outputs, indim(l) => inputs, otherdims...)
    b = select(bias(l), 1 => outputs)
    newlayer(m, w, b, otherpars(other, l))
end
otherpars(o, l) = ()

function mutate(::FluxDepthwiseConv, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m), other= l -> ())
    l = layer(m)
    otherdims = other(l)
    weightouts = map(Iterators.partition(outputs, length(inputs))) do group
        all(group .< 0) && return group[1]
        return (maximum(group) - 1) ÷ length(inputs) + 1
    end

    w = select(weights(l), outdim(l) => weightouts, indim(l) => inputs, otherdims...)
    b = select(bias(l), 1 => outputs)
    newlayer(m, w, b, otherpars(other, l))
end

function mutate(t::FluxRecurrent, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m), other=missing)
    l = layer(m)
    outputs_scaled = mapfoldl(vcat, 1:outscale(l)) do i
        offs = (i-1) * nout(l)
        return map(x -> x > 0 ? x + offs : x, outputs)
    end

    wi = select(weights(l), outdim(l) => outputs_scaled, indim(l) => inputs)
    wh = select(hiddenweights(l), 1 => outputs_scaled, 2 => outputs)
    b = select(bias(l), 1 => outputs_scaled)
    mutate_recurrent_state(t, m, outputs, wi, wh, b)
end

function mutate_recurrent_state(::FluxRecurrent, m::MutableLayer, outputs, wi, wh, b)
    l = layer(m)
    h = select(hiddenstate(l), 1 => outputs)
    s = select(state(l), 1 => outputs)

    cellnew = setproperties(layer(m).cell, (Wi=param(wi), Wh=param(wh), b = param(b), h = param(h)))
    lnew = setproperties(layer(m), (cell=cellnew, state = param(s)))
    m.layer = lnew
end

function mutate_recurrent_state(::FluxLstm, m::MutableLayer, outputs, wi, wh, b)
    l = layer(m)
    hcurr, scurr = hiddenstate(l), state(l)
    hc = select.(hcurr, repeat([1 => outputs], length(hcurr)))
    s = select.(scurr, repeat([1 => outputs], length(scurr)))

    cellnew = setproperties(layer(m).cell, (Wi=param(wi), Wh=param(wh), b = param(b), h = param(hc[1]), c = param(hc[2])))
    lnew = setproperties(layer(m), (cell=cellnew, state = tuple(param.(s)...)))
    m.layer = lnew

    return (h = hc[1], c = hc[2]), tuple(param.(s))
end


function mutate(t::FluxParInvLayer, m::MutableLayer; inputs=missing, outputs=missing, other=missing)
    @assert any(ismissing.((inputs, outputs))) || inputs == outputs "Try to mutate $inputs and $outputs for invariant layer $(m)!"
    ismissing(inputs) || return mutate(t, m, inputs)
    ismissing(outputs) || return mutate(t, m, outputs)
end

function mutate(::FluxDiagonal, m::MutableLayer, inds)
    l = layer(m)
    w = select(weights(l), 1 => inds)
    b = select(bias(l), 1 => inds)
    newlayer(m, w, b)
end

function mutate(::FluxLayerNorm, m::MutableLayer, inds)
    # LayerNorm is only a wrapped Diagonal. Just mutate the Diagonal and make a new LayerNorm of it
    proxy = MutableLayer(layer(m).diag)
    mutate(proxy, inputs=inds, outputs=inds, other=l->())
    m.layer = LayerNorm(layer(proxy))
end

function mutate(::FluxParNorm, m::MutableLayer, inds)
    # Good? bad? I'm the guy who assumes mean and std type parameters will be visited in a certain order and uses a closure for that assumption
    ismean = false
    parselect = function(x)
        ismean = !ismean
        return select(x, 1 => inds; insval = (ismean ? 0 : 1))
    end
    m.layer = Flux.mapchildren(parselect, m.layer)
end

function mutate(::FluxGroupNorm, m::MutableLayer, inds)

    l = m.layer
    ngroups = l.G
    nchannels = length(inds)
    # Make a "best fit" to new group size, but don't allow 1 as group size
    g_alts = filter(ga -> ga > 1, findall(r -> r == 0, nchannels .% (1:nchannels)))
    ngroups = g_alts[argmin(abs.(g_alts .- ngroups))]

    nchannels_per_group = div(nchannels, ngroups)
    # Map indices to (new) groups
    inds_groups = ceil.(Int, map(ind -> ind > 0 ? ind / nchannels_per_group : ind, inds))
    # TODO: Select the most commonly occuring index in each column (except -1)
    inds_groups = reshape(inds_groups, nchannels_per_group, ngroups)[1,:]

    sizetoinds = Dict(nin(l) => inds, l.G => inds_groups)

    ismean = false
    parselect = function(x)
        ismean = !ismean
        return select(x, 1 => sizetoinds[length(x)]; insval = (ismean ? 0 : 1))
    end
    m.layer = Flux.mapchildren(parselect, m.layer)
    m.layer.G = ngroups
end


newlayer(m::MutableLayer, w, b, other=nothing) = m.layer = newlayer(layertype(m), m, w, b, other)

newlayer(::FluxDense, m::MutableLayer, w, b, other) = Dense(param(w), param(b), deepcopy(layer(m).σ))
newlayer(::FluxConvolutional, m::MutableLayer, w, b, other) = setproperties(layer(m), (weight=param(w), bias=param(b), σ=deepcopy(layer(m).σ), other...))
newlayer(::FluxDiagonal, m::MutableLayer, w, b, other) = Flux.Diagonal(param(w), param(b))


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
    inputs::AbstractVector{<:Integer}
    outputs::AbstractVector{<:Integer}
    other
end
LazyMutable(m::AbstractMutableComp) = LazyMutable(m, nin(m), nout(m))
LazyMutable(m, nin::Integer, nout::Integer) = LazyMutable(m, 1:nin, 1:nout, m -> ())

wrapped(m::LazyMutable) = m.mutable
layer(m::LazyMutable) = layer(wrapped(m))

treelike_fields(T::Type{LazyMutable}) = (:mutable,)

(m::LazyMutable)(x...) = dispatch!(m, m.mutable, x...)
dispatch!(m::LazyMutable, mutable::AbstractMutableComp, x...) = mutable(x...)

NaiveNASlib.nin(m::LazyMutable) = length(m.inputs)
NaiveNASlib.nout(m::LazyMutable) = length(m.outputs)

function NaiveNASlib.mutate_inputs(m::LazyMutable, inputs::AbstractArray{<:Integer,1}...)
    @assert length(inputs) == 1 "Only one input per layer!"
    m.inputs == inputs[1] && return

    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.inputs = select(m.inputs, 1 => inputs[1], insval=-1)
end

function NaiveNASlib.mutate_outputs(m::LazyMutable, outputs::AbstractArray{<:Integer,1})
    outputs == m.outputs && return

    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.outputs = select(m.outputs, 1=>outputs, insval = -1)
end

function mutate_weights(m::LazyMutable, w)
    m.other == w && return

    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.other = w
end

NaiveNASlib.mutate_inputs(m::LazyMutable, nin::Integer...) = mutate_inputs(m, trunc_or_pad.(length(m.inputs), nin)...)

NaiveNASlib.mutate_outputs(m::LazyMutable, nout::Integer) = mutate_outputs(m, trunc_or_pad(length(m.outputs), nout))

function trunc_or_pad(maxselect, size)
    res = -ones(Int, size)
    lastselected = min(size, maxselect)
    res[1:lastselected] = 1:lastselected
    return res
end

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

function dispatch!(lm::LazyMutable, m::MutationTriggered, x...)
    mutate(m.wrapped; inputs=lm.inputs, outputs=lm.outputs, other=lm.other)
    lm.mutable = m.wrapped
    return lm(x...)
end

layer(m::MutationTriggered) = layer(m.wrapped)
layertype(m::MutationTriggered) = layertype(layer(m))

Flux.@treelike MutationTriggered

"""
    ResetLazyMutable

Reset a `LazyMutable` when dispatching.
"""
struct ResetLazyMutable
    wrapped
end
ResetLazyMutable(r::ResetLazyMutable) = r

function dispatch!(lm::LazyMutable, m::ResetLazyMutable, x...)
    lm.mutable = m.wrapped
    output = lm(x...)
    lm.inputs = 1:nin(lm)
    lm.outputs = 1:nout(lm)
    lm.other = m -> ()
    return output
end

layer(m::ResetLazyMutable) = layer(m.wrapped)
layertype(m::ResetLazyMutable) = layertype(layer(m))

Flux.@treelike ResetLazyMutable

"""
    NoParams

Ignores size mutation.

Useful for layers which don't have parameters.
"""
struct NoParams
    layer
end

(i::NoParams)(x...) = layer(i)(x...)
layer(i::NoParams) = i.layer
layertype(i::NoParams) = layertype(layer(i))

LazyMutable(m::NoParams) = m

function NaiveNASlib.mutate_inputs(::NoParams, inputs) end
function NaiveNASlib.mutate_outputs(::NoParams, outputs) end
function mutate_weights(::NoParams, w) end
NaiveNASlib.minΔninfactor(m::NoParams) = minΔninfactor(layertype(m), layer(m))
NaiveNASlib.minΔnoutfactor(m::NoParams) = minΔnoutfactor(layertype(m), layer(m))
