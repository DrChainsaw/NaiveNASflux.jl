
"""
    InputShapeVertex(v::AbstractVertex, t::FluxLayer)

Input type vertex which also has information about what type of layer the input is shaped for.
"""
struct InputShapeVertex <: AbstractVertex
    base::AbstractVertex
    t::FluxLayer
end
"""
    inputvertex(name, size, type::FluxLayer)

Return an immutable input type vertex with the given `name` and `size` and a `type` which can be used to indicate what type of input is expected.
"""
NaiveNASlib.inputvertex(name, size, type::FluxLayer) = InputShapeVertex(inputvertex(name, size), type)
layertype(v::InputShapeVertex) = v.t
layer(v::InputShapeVertex) = LayerTypeWrapper(v.t)
NaiveNASlib.base(v::InputShapeVertex) = v.base
NaiveNASlib.name(v::InputShapeVertex) = name(base(v))
NaiveNASlib.nout(v::InputShapeVertex) = nout(base(v))
NaiveNASlib.nin(v::InputShapeVertex) = nin(base(v))
NaiveNASlib.outputs(v::InputShapeVertex) = outputs(base(v))
NaiveNASlib.inputs(v::InputShapeVertex) = []
NaiveNASlib.clone(v::InputShapeVertex, ins::AbstractVertex...;cf=clone) = InputShapeVertex(cf(base(v), ins...;cf=cf), layertype(v))

# Only to prevent stack overflow above
struct LayerTypeWrapper
    t::FluxLayer
end
layertype(l::LayerTypeWrapper) = l.t

Flux.@treelike InputVertex
Flux.@treelike CompVertex

# This is a bit of a hack to enable 1) params and 2) gpu. Other uses may not work as expected, especially if one tries to use these methods to view/manipulate things which are not from Flux.

# Two things (afaik) prevent usage of @treelike:
#   1) Flux issue #803 which is fixed on master but not on any release
#   2) MutationVertices (OutputVertices really) can not be created by just copying their fields due to the cyclic structure (or?) and mapchildren seems to be designed around this.

# Instead, we rely in the internals of the vertices to be mutable (e.g MutableLayer).

Flux.children(a::AbstractVector{<:AbstractVertex}) = Tuple(a)
function Flux.mapchildren(f, a::AbstractVector{<:AbstractVertex})
    f.(a) # Returning this will do no good due to 2) above
    return a
end

Flux.children(v::AbstractVertex) = (base(v),)
function Flux.mapchildren(f, v::AbstractVertex)
    f.(Flux.children(v)) # Returning this will do no good due to 2) above
    return v
end
Flux.children(g::CompGraph) = Tuple(vertices(g))
function Flux.mapchildren(f, g::CompGraph)
    f.(Flux.children(g)) # Returning this will do no good due to 2) above
    return g
end


"""
    mutable(l, in::AbstractVertex; layerfun=LazyMutable, mutation=IoChange, traitfun=validated())

Return a mutable vertex wrapping the layer `l` with input vertex `in`.

Extra arguments `layerfun`, `mutation` and `traitfun` can be used to change mutation type and to add extra info about the vertex.
"""
mutable(l, in::AbstractVertex; layerfun=LazyMutable, mutation=IoChange, traitfun=validated()) = mutable(layertype(l), l, in, layerfun, mutation, traitfun)

"""
    mutable(name::String, l, in::AbstractVertex; layerfun=LazyMutable, mutation=IoChange, traitfun=validated())

Return a mutable vertex wrapping the layer `l` with input vertex `in` with name `name`.

Name is only used when displaying or logging and does not have to be unique (although it probably is a good idea).

Extra arguments `layerfun`, `mutation` and `traitfun` can be used to change mutation type and to add extra info about the vertex.
"""
mutable(name::String, l, in::AbstractVertex; layerfun=LazyMutable, mutation=IoChange, traitfun=validated()) = mutable(layertype(l), l, in, layerfun, mutation, traitfun ∘ named(name))

mutable(::FluxParLayer, l, in::AbstractVertex, layerfun, mutation, traitfun) = absorbvertex(layerfun(MutableLayer(l)), nout(l), in, mutation=mutation, traitdecoration = traitfun)

mutable(::FluxParInvLayer, l, in::AbstractVertex, layerfun, mutation, traitfun) =
invariantvertex(layerfun(MutableLayer(l)), in, mutation=mutation, traitdecoration=traitfun)

mutable(::FluxNoParLayer, l, in::AbstractVertex, layerfun, mutation, traitfun) = invariantvertex(layerfun(NoParams(l)), in, mutation=mutation, traitdecoration=traitfun)

# Decorate trait with extra stuff like logging of size changes or validation.
# Meant to be composable, e.g. using ∘
named(name) = t -> NamedTrait(t, name)
validated() = t -> SizeChangeValidation(t)
logged(;level = Base.CoreLogging.Debug, info = FullInfoStr()) = t -> SizeChangeLogger(level, info, t)

"""
   concat(v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitfun=identity)

Return a mutable vertex which concatenates input.

Inputs must have compatible activation shapes or an exception will be thrown.

Extra arguments `layerfun`, `mutation` and `traitfun` can be used to change mutation type and to add extra info about the vertex.
"""
function concat(v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitfun=identity, layerfun=identity)
    dims = tuple(Iterators.flatten(actdim.([v, vs...]))...)
    ranks = tuple(Iterators.flatten(actrank.([v, vs...]))...)
    concat(Val.(dims), Val.(ranks), mutation, traitfun, layerfun, v, vs...)
end

"""
    concat(name::String, v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitfun=identity)

Return a mutable vertex with name `name` which concatenates input.

Name is only used when displaying or logging and does not have to be unique (although it probably is a good idea).

Inputs must have compatible activation shapes or an exception will be thrown.

Extra arguments `layerfun`, `mutation` and `traitfun` can be used to change mutation type and to add extra info about the vertex.
"""
concat(name::String, v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitfun = identity, layerfun=identity) = concat(v, vs..., mutation=mutation, traitfun=traitfun ∘ named(name), layerfun=layerfun)

concat(actdims, actranks, mutation, traitfun, layerfun, v::AbstractVertex, vs::AbstractVertex...) = throw(DimensionMismatch("Can not concatenate activations with different shapes! Got: $actdims and $actranks")) # I guess it might be doable, but CBA to try it out

# NTuples only match if all actdims (N) and actranks (M) are identical
# Can't ... stop ... dispatching ... on ... stuff ...
concat(::NTuple{T, Val{N}}, ::NTuple{T, Val{M}}, mutation, traitfun, layerfun, v::AbstractVertex, vs::AbstractVertex...) where {T,N,M} = conc(v, vs..., dims=N, mutation=mutation, traitdecoration=traitfun, outwrap=layerfun)


layer(v::AbstractVertex) = layer(base(v))
layer(v::CompVertex) = layer(v.computation)

layertype(v::AbstractVertex) = layertype(base(v))
layertype(v::CompVertex) = layertype(v.computation)

actdim(v::AbstractVertex) = actdim.(layer.(NaiveNASlib.findterminating(v, inputs)))
actrank(v::AbstractVertex) = actrank.(layer.(NaiveNASlib.findterminating(v, inputs)))
