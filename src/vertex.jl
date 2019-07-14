
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
NaiveNASlib.nout(v::InputShapeVertex) = nout(base(v))
NaiveNASlib.nin(v::InputShapeVertex) = nin(base(v))
NaiveNASlib.outputs(v::InputShapeVertex) = outputs(base(v))
NaiveNASlib.inputs(v::InputShapeVertex) = []
NaiveNASlib.clone(v::InputShapeVertex, ins::AbstractVertex...) = clone(base(v), ins...)

# Only to prevent stack overflow above
struct LayerTypeWrapper
    t::FluxLayer
end
layertype(l::LayerTypeWrapper) = l.t

Flux.@treelike CompGraph
Flux.children(a::AbstractVector{<:AbstractVertex}) = Tuple(a)
# Avoid outputs due to flux issue #803
Flux.children(v::AbstractVertex) = tuple(base(v), inputs(v))
Flux.@treelike InputVertex
Flux.@treelike CompVertex


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
   concat(v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitdecoration=identity)

Return a mutable vertex which concatenates input.

Inputs must have compatible activation shapes or an exception will be thrown.
"""
function concat(v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitdecoration=identity)
    dims = tuple(Iterators.flatten(actdim.([v, vs...]))...)
    ranks = tuple(Iterators.flatten(actrank.([v, vs...]))...)
    concat(Val.(dims), Val.(ranks), mutation, traitdecoration, v, vs...)
end

concat(actdims, actranks, mutation, traitdecoration, v::AbstractVertex, vs::AbstractVertex...) = throw(DimensionMismatch("Can not concatenate activations with different shapes! Got: $actdims and $actranks")) # I guess it might be doable, but CBA to try it out

# NTuples only match if all actdims (N) and actranks (M) are identical
# Can't ... stop ... dispatching ... on ... stuff ...
concat(::NTuple{T, Val{N}}, ::NTuple{T, Val{M}}, mutation, traitdecoration, v::AbstractVertex, vs::AbstractVertex...) where {T,N,M} = conc(v, vs..., dims=N, mutation=mutation, traitdecoration=traitdecoration)


layer(v::AbstractVertex) = layer(base(v))
layer(v::CompVertex) = layer(v.computation)

layertype(v::AbstractVertex) = layertype(base(v))
layertype(v::CompVertex) = layertype(v.computation)

actdim(v::AbstractVertex) = actdim.(layer.(NaiveNASlib.findterminating(v, inputs)))
actrank(v::AbstractVertex) = actrank.(layer.(NaiveNASlib.findterminating(v, inputs)))
