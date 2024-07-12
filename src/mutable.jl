
"""
     AbstractMutableComp

Abstract base type for computation units (read layers) which may mutate.
"""
abstract type AbstractMutableComp end
Base.Broadcast.broadcastable(m::AbstractMutableComp) = Ref(m)

# Generic forwarding methods. Just implement layer(t::Type) and wrapped(t::Type) to enable

# Not possible in julia <= 1.1. See #14919
# (m::AbstractMutableComp)(x...) = layer(m)(x...)
layer(m::AbstractMutableComp) = layer(wrapped(m))
layertype(m::AbstractMutableComp) = layertype(layer(m))
NaiveNASlib.op(m::AbstractMutableComp) = layer(m)

NaiveNASlib.nin(m::AbstractMutableComp) = nin(wrapped(m))
NaiveNASlib.nout(m::AbstractMutableComp) = nout(wrapped(m))

function NaiveNASlib.Δsize!(m::AbstractMutableComp, inputs::AbstractVector, outputs::AbstractVector;kwargs...) 
     NaiveNASlib.Δsize!(wrapped(m), inputs, outputs;kwargs...)
end

mutate_weights(m::AbstractMutableComp, w) = mutate_weights(wrapped(m), w)

function NaiveNASlib.compconstraint!(case, s::AbstractJuMPΔSizeStrategy, m::AbstractMutableComp, data) 
     NaiveNASlib.compconstraint!(case, s, layertype(m), data)
end

function NaiveNASlib.defaultutility(m::AbstractMutableComp) 
    util = neuronutility_safe(m)
    return util === 1 ? NaiveNASlib.NoDefaultUtility() : util
end

"""
    MutableLayer

Wraps a layer in order to allow for mutation as layer structures are typically immutable
"""
mutable struct MutableLayer <: AbstractMutableComp
    layer # Can't specialize as this might mutate into something else, e.g. Dense{Float32} -> Dense{Float64} thanks to mutable functor. Try to make functor rebuild everything?
end
(m::MutableLayer)(x...) = layer(m)(x...)
wrapped(m::MutableLayer) = m.layer
layer(m::MutableLayer) = wrapped(m)
layertype(m::MutableLayer) = layertype(layer(m))

@layer :expand MutableLayer


function NaiveNASlib.Δsize!(m::MutableLayer, inputs::AbstractVector, outputs::AbstractVector; insert=neuroninsert, kwargs...) 
    @assert length(inputs) == 1 "Only one input per layer!"
    mutate(m; inputs=inputs[1], outputs=outputs, insert=insert, kwargs...)
    nothing
end

mutate_weights(m::MutableLayer, w) = mutate(layertype(m), m, other=w)

function mutate(m::MutableLayer; inputs, outputs, other = l -> (), insert=neuroninsert) 
    if inputs === missing
        mutate(layertype(m), m; outputs, other, insert)
    else
        mutate(layertype(m), m; inputs, outputs, other, insert)
    end
end

mutate(lt::FluxParLayer, m::MutableLayer; kwargs...) = _mutate(lt, m; kwargs...)

function _mutate(lt::FluxParLayer, m::MutableLayer; inputs=1:nin(m)[], outputs=1:nout(m), other= l -> (), insert=neuroninsert)
    l = layer(m)
    otherdims = other(l)
    w = select(weights(l), indim(l) => inputs, outdim(l) => outputs, otherdims...; newfun=insert(lt, WeightParam()))
    b = select(bias(l), 1 => outputs; newfun=insert(lt, BiasParam()))
    newlayer(m, w, b, otherpars(other, l))
end
otherpars(o, l) = ()

function mutate(lt::FluxConvolutional{N}, m::MutableLayer; inputs=1:nin(m)[], outputs=1:nout(m), other= l -> (), insert=neuroninsert) where N

    if ngroups(lt, layer(m)) == 1
        return _mutate(lt, m; inputs, outputs, other, insert)
    end

    l = layer(m)
    otherdims = other(l)

    # TODO: Handle other cases than ngroups == nin
    newingroups = 1

    # inputs and outputs are coupled through the constraints (which hopefully were enforced) so we only need to consider outputs
    currsize =size(weights(l))
    wo = select(reshape(weights(l), currsize[1:N]...,:), N+1 => outputs, otherdims...; newfun=insert(lt, WeightParam()))
    newks = size(wo)[1:N]
    w = collect(reshape(wo, newks...,newingroups, :))
    b = select(bias(l), 1 => outputs; newfun=insert(lt, BiasParam()))
    newlayer(m, w, b, (;groups= length(inputs) ÷ newingroups, otherpars(other, l)...))
end

function mutate(lt::FluxRecurrent, m::MutableLayer; inputs=1:nin(m)[], outputs=1:nout(m), other=missing, insert=neuroninsert)
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
    state0 = select(hiddenstate(l), 1 => outputs, newfun=insert(lt, RecurrentState()))
    s = select(state(l), 1 => outputs; newfun=insert(lt, RecurrentState()))

    cellnew = setproperties(layer(m).cell, (Wi=wi, Wh=wh, b = b, state0 = state0))
    lnew = setproperties(layer(m), (cell=cellnew, state = s))
    m.layer = lnew
end

function mutate_recurrent_state(lt::FluxLstm, m::MutableLayer, outputs, wi, wh, b, insert)
    l = layer(m)
    s0curr, scurr = hiddenstate(l), state(l)
    s0 = select.(s0curr, repeat([1 => outputs], length(s0curr)); newfun=insert(lt, RecurrentState()))
    s = select.(scurr, repeat([1 => outputs], length(scurr)); newfun=insert(lt, RecurrentState()))


    cellnew = setproperties(layer(m).cell, (Wi=wi, Wh=wh, b = b, state0 = Tuple(s0)))
    lnew = setproperties(layer(m), (cell=cellnew, state = Tuple(s)))
    m.layer = lnew

    return (;state0 = Tuple(s0)), Tuple(s)
end


function mutate(t::FluxParInvLayer, m::MutableLayer; inputs=missing, outputs=missing, other=missing, insert=neuroninsert)
    @assert any(ismissing.((inputs, outputs))) || inputs == outputs "Try to mutate $inputs and $outputs for invariant layer $(m)!"
    ismissing(inputs) || return mutate(t, m, inputs; insert=insert)
    ismissing(outputs) || return mutate(t, m, outputs; insert=insert)
end

function mutate(lt::FluxScale, m::MutableLayer, inds; insert=neuroninsert)
    l = layer(m)
    w = select(weights(l), 1 => inds, newfun=insert(lt, WeightParam()))
    b = select(bias(l), 1 => inds; newfun=insert(lt, BiasParam()))
    newlayer(m, w, b)
end

function mutate(::FluxLayerNorm, m::MutableLayer, inds; insert=neuroninsert)
    # LayerNorm is only a wrapped Scale. Just mutate the Scale and make a new LayerNorm of it
    proxy = MutableLayer(layer(m).diag)
    mutate(proxy; inputs=inds, outputs=inds, other=l->(), insert=insert)

    updatelayer = layer(m)
    m.layer = @set updatelayer.diag = layer(proxy)
end

function mutate(lt::FluxParNorm, m::MutableLayer, inds; insert=neuroninsert)

    # Filter out the parameters which need to change and decide for each name (e.g. γ, β etc) what to do (typically insert 1 for scaling things and 0 for offset things)
    parselect(p::Pair) = parselect(p...)
    parselect(pname, x::AbstractArray) = select(x, 1 => inds; newfun = neuroninsert(lt, pname))
    parselect(pname, x) = x

    fs, re = Flux.functor(m.layer)
    newlayer = re(map(parselect, pairs(fs) |> collect))
    newlayer = @set newlayer.chs = length(inds)
    m.layer = newlayer
end

function mutate(lt::FluxGroupNorm, m::MutableLayer, inds; insert=neuroninsert)

    # TODO: This alg was from when Flux used a different way to represent the parameters of GroupNorm
    # For some reason it still produces correct results in testcases, but it might very well do so for
    # the wrong reasons...

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

    sizetoinds = Dict(nin(l)[] => inds, l.G => inds_groups)

    parselect(p::Pair) = parselect(p...)
    parselect(pname, x::AbstractArray) = select(x, 1 => sizetoinds[length(x)]; newfun = insert(lt, pname))
    parselect(pname, x) = x

    fs, re = Flux.functor(m.layer)
    newlayer = re(map(parselect, pairs(fs) |> collect))
    newlayer.G = ngroups
    newlayer = @set newlayer.chs = length(inds)
    m.layer = newlayer
    m.layer.G = ngroups
end


newlayer(m::MutableLayer, w, b, other=nothing) = m.layer = newlayer(layertype(m), m, w, b, other)

newlayer(::FluxDense, m::MutableLayer, w, b, other) = Dense(w, b, deepcopy(layer(m).σ))
newlayer(::FluxConvolutional, m::MutableLayer, w, b, other) = setproperties(layer(m), (weight=w, bias=b, σ=deepcopy(layer(m).σ), other...))
newlayer(::FluxScale, m::MutableLayer, w, b, other) = Flux.Scale(w, b)


"""
    LazyMutable
    LazyMutable(m::AbstractMutableComp)

Lazy version of MutableLayer in the sense that it does not perform any mutations
until invoked to perform a computation.

This reduces the need to garbage collect when multiple mutations might be applied
to a vertex before evaluating the model.

Also useable for factory-like designs where the actual layers of a computation graph
are not instantiated until the graph is used.

# Examples
```jldoctest
julia> using NaiveNASflux, Flux

julia> struct DenseConfig end

julia> lazy = LazyMutable(DenseConfig(), 2, 3);

julia> layer(lazy)
DenseConfig()

julia> function NaiveNASflux.dispatch!(m::LazyMutable, ::DenseConfig, x)
       m.mutable = Dense(nin(m)[1], nout(m), relu)
       return m.mutable(x)
       end;

julia> lazy(ones(Float32, 2, 5)) |> size
(3, 5)

julia> layer(lazy)
Dense(2 => 3, relu)  # 9 parameters
```
"""
mutable struct LazyMutable <: AbstractMutableComp
    mutable # May change at any time (see example)
    inputs::Vector{Vector{Int}}
    outputs::Vector{Int}
    other # May change at any time
    insert # May change at any time
end
LazyMutable(m::AbstractMutableComp) = LazyMutable(m, nin(m), nout(m))
LazyMutable(m, nin::Integer, nout::Integer) = LazyMutable(m, [nin], nout)
function LazyMutable(m, nins::AbstractVector{<:Integer}, nout::Integer)  
    inputs = map(nin -> collect(Int, 1:nin), nins)
    outputs = collect(Int, 1:nout)
    LazyMutable(m, inputs, outputs, m -> (), neuroninsert)
end

function Base.show(io::IO, lm::LazyMutable)
    print(io, "LazyMutable(")
    show(io, lm.mutable)
    print(io, '[')
    for inpt in lm.inputs
        print(io, NaiveNASlib.compressed_string(inpt))
    end
    print(io, ']', NaiveNASlib.compressed_string(lm.outputs))
    show(io, lm.other)
    show(io, lm.insert)
end

wrapped(m::LazyMutable) = m.mutable
layer(m::LazyMutable) = layer(wrapped(m))

function Functors.functor(::Type{<:LazyMutable}, m)
    forcemutation(m)
    return (mutable=m.mutable,), y -> LazyMutable(y.mutable)
end

(m::LazyMutable)(x...) = dispatch!(m, m.mutable, x...)
dispatch!(::LazyMutable, m::AbstractMutableComp, x...) = m(x...)

NaiveNASlib.nin(m::LazyMutable) = length.(m.inputs)
NaiveNASlib.nout(m::LazyMutable) = length(m.outputs)

function NaiveNASlib.Δsize!(m::LazyMutable, inputs::AbstractVector, outputs::AbstractVector; insert=neuroninsert)
    (all((i,k)::Tuple -> ismissing(i) || i == 1:k, zip(inputs, nin(m))) && outputs == 1:nout(m)) && return
    m.insert = insert
    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.inputs = map(m.inputs, inputs) do currins, newins
        ismissing(newins) && return currins
        select(currins, 1 => newins, newfun = (args...) -> -1)
    end
    m.outputs = select(m.outputs, 1=>outputs, newfun = (args...) -> -1)
    nothing
end

function mutate_weights(m::LazyMutable, w)
    m.other == w && return

    m.mutable = ResetLazyMutable(trigger_mutation(m.mutable))
    m.other = w
end

function forcemutation(x) end
function forcemutation(::InputVertex) end
forcemutation(g::CompGraph) = forcemutation.(vertices(g::CompGraph))
forcemutation(v::AbstractVertex) = forcemutation(base(v))
forcemutation(v::CompVertex) = forcemutation(v.computation)
forcemutation(m::AbstractMutableComp) = forcemutation(NaiveNASflux.wrapped(m))
forcemutation(m::LazyMutable) = m(NoComp())

struct NoComp end
function (::MutableLayer)(::NoComp) end

"""
    MutationTriggered

Dispatches mutation for LazyMutable.
"""
struct MutationTriggered{T}
    wrapped::T
end
# Functionality is opt-in
trigger_mutation(m) = m
trigger_mutation(m::AbstractMutableComp) = MutationTriggered(m)

function dispatch!(lm::LazyMutable, m::MutationTriggered, x...)
    NaiveNASlib.Δsize!(m.wrapped, lm.inputs, lm.outputs; other=lm.other, insert=lm.insert)
    lm.mutable = m.wrapped
    return lm(x...)
end

layer(m::MutationTriggered) = layer(m.wrapped)
layertype(m::MutationTriggered) = layertype(layer(m))

@layer :expand MutationTriggered

"""
    ResetLazyMutable

Reset a `LazyMutable` when dispatching.
"""
struct ResetLazyMutable{T}
    wrapped::T
end
ResetLazyMutable(r::ResetLazyMutable) = r

function dispatch!(lm::LazyMutable, m::ResetLazyMutable, x...)
    lm.mutable = m.wrapped
    output = lm(x...)
    lm.inputs = map(i -> collect(1:i), nin(lm))
    lm.outputs = 1:nout(lm)
    lm.other = m -> ()
    lm.insert = neuroninsert
    return output
end

layer(m::ResetLazyMutable) = layer(m.wrapped)
layertype(m::ResetLazyMutable) = layertype(layer(m))

@layer :expand ResetLazyMutable

"""
    NoParams

Ignores size mutation.

Useful for layers which don't have parameters.
"""
struct NoParams{T}
    layer::T
end

(i::NoParams)(x...) = layer(i)(x...)
layer(i::NoParams) = i.layer
layertype(i::NoParams) = layertype(layer(i))
NaiveNASlib.op(i::NoParams) = layer(i)

LazyMutable(m::NoParams) = m

function mutate_weights(::NoParams, w) end