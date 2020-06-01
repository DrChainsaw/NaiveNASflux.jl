
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

NaiveNASlib.mutate_inputs(m::AbstractMutableComp, inputs::AbstractArray{<:Integer,1}...;kwargs...) = mutate_inputs(wrapped(m), inputs...;kwargs...)
NaiveNASlib.mutate_outputs(m::AbstractMutableComp, outputs; kwargs...) = mutate_outputs(wrapped(m), outputs; kwargs...)

mutate_weights(m::AbstractMutableComp, w) = mutate_weights(wrapped(m), w)

NaiveNASlib.minΔninfactor(m::AbstractMutableComp) = minΔninfactor(layertype(m), layer(m))
NaiveNASlib.minΔnoutfactor(m::AbstractMutableComp) = minΔnoutfactor(layertype(m), layer(m))

NaiveNASlib.compconstraint!(s, m::AbstractMutableComp, data) = NaiveNASlib.compconstraint!(s, layertype(m), data)

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

function NaiveNASlib.mutate_inputs(m::MutableLayer, inputs::AbstractArray{<:Integer,1}...; insert=neuroninsert)
    @assert length(inputs) == 1 "Only one input per layer!"
    mutate(layertype(m), m; inputs=inputs[1], insert=insert)
end

NaiveNASlib.mutate_outputs(m::MutableLayer, outputs; insert=neuroninsert) = mutate(layertype(m), m; outputs=outputs, insert=insert)

mutate_weights(m::MutableLayer, w) = mutate(layertype(m), m, other=w)

mutate(m::MutableLayer; inputs, outputs, other = l -> (), insert=neuroninsert) = mutate(layertype(m), m, inputs=inputs, outputs=outputs, other=other, insert=insert)

function mutate(lt::FluxParLayer, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m), other= l -> (), insert=neuroninsert)
    l = layer(m)
    otherdims = other(l)
    w = select(weights(l), indim(l) => inputs, outdim(l) => outputs, otherdims...; newfun=insert(lt, WeightParam()))
    b = select(bias(l), 1 => outputs; newfun=insert(lt, BiasParam()))
    newlayer(m, w, b, otherpars(other, l))
end
otherpars(o, l) = ()

function mutate(lt::FluxDepthwiseConv, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m), other= l -> (), insert=neuroninsert)
    l = layer(m)
    otherdims = other(l)
    weightouts = map(Iterators.partition(outputs, length(inputs))) do group
        all(group .< 0) && return group[1]
        return (maximum(group) - 1) ÷ length(inputs) + 1
    end

    w = select(weights(l), indim(l) => inputs, outdim(l) => weightouts, otherdims...; newfun=insert(lt, WeightParam()))
    b = select(bias(l), 1 => outputs; newfun=insert(lt, BiasParam()))
    newlayer(m, w, b, otherpars(other, l))
end

function mutate(lt::FluxRecurrent, m::MutableLayer; inputs=1:nin(m), outputs=1:nout(m), other=missing, insert=neuroninsert)
    l = layer(m)
    outputs_scaled = mapfoldl(vcat, 1:outscale(l)) do i
        offs = (i-1) * nout(l)
        return map(x -> x > 0 ? x + offs : x, outputs)
    end

    wi = select(weights(l), indim(l) => inputs, outdim(l) => outputs_scaled, newfun=insert(lt, WeightParam()))
    wh = select(hiddenweights(l), 2 => outputs, 1 => outputs_scaled, newfun=insert(lt, RecurrentWeightParam()))
    b = select(bias(l), 1 => outputs_scaled, newfun=insert(lt, BiasParam()))
    mutate_recurrent_state(lt, m, outputs, wi, wh, b, insert)
end

function mutate_recurrent_state(lt::FluxRecurrent, m::MutableLayer, outputs, wi, wh, b, insert)
    l = layer(m)
    h = select(hiddenstate(l), 1 => outputs, newfun=insert(lt, RecurrentState()))
    s = select(state(l), 1 => outputs; newfun=insert(lt, RecurrentState()))

    cellnew = setproperties(layer(m).cell, (Wi=wi, Wh=wh, b = b, h = h))
    lnew = setproperties(layer(m), (cell=cellnew, state = s))
    m.layer = lnew
end

function mutate_recurrent_state(lt::FluxLstm, m::MutableLayer, outputs, wi, wh, b, insert)
    l = layer(m)
    hcurr, scurr = hiddenstate(l), state(l)
    hc = select.(hcurr, repeat([1 => outputs], length(hcurr)); newfun=insert(lt, RecurrentState()))
    s = select.(scurr, repeat([1 => outputs], length(scurr)); newfun=insert(lt, RecurrentState()))

    cellnew = setproperties(layer(m).cell, (Wi=wi, Wh=wh, b = b, h = hc[1], c = hc[2]))
    lnew = setproperties(layer(m), (cell=cellnew, state = tuple(s...)))
    m.layer = lnew

    return (h = hc[1], c = hc[2]), tuple(s)
end


function mutate(t::FluxParInvLayer, m::MutableLayer; inputs=missing, outputs=missing, other=missing, insert=neuroninsert)
    @assert any(ismissing.((inputs, outputs))) || inputs == outputs "Try to mutate $inputs and $outputs for invariant layer $(m)!"
    ismissing(inputs) || return mutate(t, m, inputs; insert=insert)
    ismissing(outputs) || return mutate(t, m, outputs; insert=insert)
end

function mutate(lt::FluxDiagonal, m::MutableLayer, inds; insert=neuroninsert)
    l = layer(m)
    w = select(weights(l), 1 => inds, newfun=insert(lt, WeightParam()))
    b = select(bias(l), 1 => inds; newfun=insert(lt, BiasParam()))
    newlayer(m, w, b)
end

function mutate(::FluxLayerNorm, m::MutableLayer, inds; insert=neuroninsert)
    # LayerNorm is only a wrapped Diagonal. Just mutate the Diagonal and make a new LayerNorm of it
    proxy = MutableLayer(layer(m).diag)
    mutate(proxy; inputs=inds, outputs=inds, other=l->(), insert=insert)
    m.layer = LayerNorm(layer(proxy))
end

function mutate(lt::FluxParNorm, m::MutableLayer, inds; insert=neuroninsert)

    # Filter out the parameters which need to change and decide for each name (e.g. γ, β etc) what to do (typically insert 1 for scaling things and 0 for offset things)
    parselect(p::Pair) = parselect(p...)
    parselect(pname, x::AbstractArray) = select(x, 1 => inds; newfun = neuroninsert(lt, pname))
    parselect(pname, x) = x

    fs, re = Flux.functor(m.layer)
    m.layer = re(map(parselect, pairs(fs) |> collect))
end

function mutate(lt::FluxGroupNorm, m::MutableLayer, inds; insert=neuroninsert)

    l = m.layer
    ngroups = l.G
    nchannels = length(inds)

    # Make a "best fit" to new group size, but don't allow 1 as group size
    # Step 1: Possible group sizes are all integers which divde the new number of channels evenly
    g_alts = filter(ga -> ga > 1, findall(r -> r == 0, nchannels .% (1:nchannels)))
    # Step 2: Select the size which is closest to the original number of groups
    ngroups = g_alts[argmin(abs.(g_alts .- ngroups))]

    nchannels_per_group = div(nchannels, ngroups)
    # Map indices to (new) groups
    inds_groups = ceil.(Int, map(ind -> ind > 0 ? ind / nchannels_per_group : ind, inds))
    # TODO: Select the most commonly occuring index in each column (except -1)
    inds_groups = reshape(inds_groups, nchannels_per_group, ngroups)[1,:]

    sizetoinds = Dict(nin(l) => inds, l.G => inds_groups)

    parselect(p::Pair) = parselect(p...)
    parselect(pname, x::AbstractArray) = select(x, 1 => sizetoinds[length(x)]; newfun = insert(lt, pname))
    parselect(pname, x) = x

    fs, re = Flux.functor(m.layer)
    m.layer = re(map(parselect, pairs(fs) |> collect))
    m.layer.G = ngroups
end


newlayer(m::MutableLayer, w, b, other=nothing) = m.layer = newlayer(layertype(m), m, w, b, other)

newlayer(::FluxDense, m::MutableLayer, w, b, other) = Dense(w, b, deepcopy(layer(m).σ))
newlayer(::FluxConvolutional, m::MutableLayer, w, b, other) = setproperties(layer(m), (weight=w, bias=b, σ=deepcopy(layer(m).σ), other...))
newlayer(::FluxDiagonal, m::MutableLayer, w, b, other) = Flux.Diagonal(w, b)


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
    insert
end
LazyMutable(m::AbstractMutableComp) = LazyMutable(m, nin(m), nout(m))
LazyMutable(m, nin::Integer, nout::Integer) = LazyMutable(m, 1:nin, 1:nout, m -> (), neuroninsert)

wrapped(m::LazyMutable) = m.mutable
layer(m::LazyMutable) = layer(wrapped(m))

function Flux.functor(m::LazyMutable)
    forcemutation(m)
    return (mutable=m.mutable,),
    function(y)
        m.mutable = y[1]
        return m
    end
end

(m::LazyMutable)(x...) = dispatch!(m, m.mutable, x...)
dispatch!(m::LazyMutable, mutable::AbstractMutableComp, x...) = mutable(x...)

NaiveNASlib.nin(m::LazyMutable) = length(m.inputs)
NaiveNASlib.nout(m::LazyMutable) = length(m.outputs)

function NaiveNASlib.mutate_inputs(m::LazyMutable, inputs::AbstractArray{<:Integer,1}...; insert=neuroninsert)
    @assert length(inputs) == 1 "Only one input per layer!"
    m.inputs == inputs[1] && return

    m.insert = insert
    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.inputs = select(m.inputs, 1 => inputs[1], newfun = (args...) -> -1)
end

function NaiveNASlib.mutate_outputs(m::LazyMutable, outputs::AbstractArray{<:Integer,1}; insert=neuroninsert)
    outputs == m.outputs && return

    m.insert = insert
    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.outputs = select(m.outputs, 1=>outputs, newfun = (args...) -> -1)
end

function mutate_weights(m::LazyMutable, w)
    m.other == w && return

    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.other = w
end

NaiveNASlib.mutate_inputs(m::LazyMutable, nin::Integer...; insert=neuroninsert) = mutate_inputs(m, trunc_or_pad.(length(m.inputs), nin)...;insert=insert)

NaiveNASlib.mutate_outputs(m::LazyMutable, nout::Integer; insert=neuroninsert) = mutate_outputs(m, trunc_or_pad(length(m.outputs), nout); insert=insert)

function trunc_or_pad(maxselect, size)
    res = -ones(Int, size)
    lastselected = min(size, maxselect)
    res[1:lastselected] = 1:lastselected
    return res
end

function forcemutation(x) end
function forcemutation(v::InputVertex) end
forcemutation(g::CompGraph) = forcemutation.(vertices(g::CompGraph))
forcemutation(v::AbstractVertex) = forcemutation(base(v))
forcemutation(v::CompVertex) = forcemutation(v.computation)
forcemutation(m::AbstractMutableComp) = forcemutation(NaiveNASflux.wrapped(m))
forcemutation(m::LazyMutable) = m(NoComp())

struct NoComp end
function (::MutableLayer)(x::NoComp) end

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
    mutate(m.wrapped; inputs=lm.inputs, outputs=lm.outputs, other=lm.other, insert=lm.insert)
    lm.mutable = m.wrapped
    return lm(x...)
end

layer(m::MutationTriggered) = layer(m.wrapped)
layertype(m::MutationTriggered) = layertype(layer(m))

Flux.@functor MutationTriggered

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
    lm.insert = neuroninsert
    return output
end

layer(m::ResetLazyMutable) = layer(m.wrapped)
layertype(m::ResetLazyMutable) = layertype(layer(m))

Flux.@functor ResetLazyMutable

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

function NaiveNASlib.mutate_inputs(::NoParams, inputs;insert=missing) end
function NaiveNASlib.mutate_outputs(::NoParams, outputs;insert=missing) end
function mutate_weights(::NoParams, w) end
NaiveNASlib.minΔninfactor(m::NoParams) = minΔninfactor(layertype(m), layer(m))
NaiveNASlib.minΔnoutfactor(m::NoParams) = minΔnoutfactor(layertype(m), layer(m))
