
"""
    InputShapeVertex <: AbstractVertex
    InputShapeVertex(v::AbstractVertex, t::FluxLayer)

Input type vertex which also has information about what type of layer the input is shaped for.
"""
struct InputShapeVertex{V<:AbstractVertex, L<:FluxLayer} <: AbstractVertex
    base::V
    t::L
end
"""
    inputvertex(name, size, type::FluxLayer)

Return an immutable input type vertex with the given `name` and `size` and a `type` which can be used to 
indicate what type of input is expected.
"""
NaiveNASlib.inputvertex(name, size, type::FluxLayer) = InputShapeVertex(inputvertex(name, size), type)

"""
    convinputvertex(name, size, ndims) 

Return an input type vertex with the given `name` which promises convolution shaped input 
with `size` channels and `ndims` number of dimensions for feature maps (e.g. 2 for images)
suitable for `Flux`s convolution layers.
"""
convinputvertex(name, size, ndims) = inputvertex(name, size, FluxConv{ndims}())

"""
    denseinputvertex(name, size)

Return an input type vertex with the given `name` which promises 2D shaped input
with `size` number of features suitable for e.g. `Flux`s `Dense` layer.
"""
denseinputvertex(name, size) = inputvertex(name, size, FluxDense())

"""
    rnninputvertex(name, size)

Return an input type vertex with the given `name` which promises 2D shaped input
with `size` number of features suitable for `Flux`s recurrent layers.
"""
rnninputvertex(name, size) = inputvertex(name, size, FluxRnn())

layertype(v::InputShapeVertex) = v.t
layer(v::InputShapeVertex) = LayerTypeWrapper(v.t) # so that layertype(layer(v)) works
NaiveNASlib.base(v::InputShapeVertex) = v.base
NaiveNASlib.name(v::InputShapeVertex) = name(base(v))
NaiveNASlib.nout(v::InputShapeVertex) = nout(base(v))
NaiveNASlib.nin(v::InputShapeVertex) = nin(base(v))
NaiveNASlib.outputs(v::InputShapeVertex) = outputs(base(v))
NaiveNASlib.inputs(::InputShapeVertex) = []
NaiveNASlib.clone(v::InputShapeVertex, ins::AbstractVertex...;cf=clone) = InputShapeVertex(cf(base(v), ins...;cf=cf), layertype(v))

# Only to prevent stack overflow above
struct LayerTypeWrapper{L}
    t::L
end
layertype(l::LayerTypeWrapper) = l.t

"""
    SizeNinNoutConnected <: NaiveNASlib.DecoratingTrait
    SizeNinNoutConnected(t)

Trait for computations for which a change in output size results in a change in input size but which 
is not fully `SizeTransparent`.

Example of this is DepthWiseConv where output size must be an integer multiple of the input size.

Does not create any constraints or objectives, only signals that vertices after a 
`SizeNinNoutConnected` might need to change size if the size of the `SizeNinNoutConnected` vertex changes.
"""
struct SizeNinNoutConnected{T <: NaiveNASlib.MutationTrait} <: NaiveNASlib.DecoratingTrait
    base::T
end
NaiveNASlib.base(t::SizeNinNoutConnected) = t.base


NaiveNASlib.all_in_Δsize_graph(::SizeNinNoutConnected, d, v, visited) = all_in_Δsize_graph(SizeInvariant(), d, v, visited)


Flux.@functor InputVertex
Flux.@functor CompVertex

# This is a bit of a hack to enable 1) params and 2) gpu. Other uses may not work as expected, especially if one tries to use these methods to view/manipulate things which are not from Flux.

# Problem with using Flux.@functor is that MutationVertices (OutputVertices really) can not be created by just copying their fields as this would create multiple copies of the same vertex if it is input to more than one vertex.
# Instead, we rely in the internals of the vertices to be mutable (e.g MutableLayer).

Flux.functor(::Type{<:AbstractVector{<:AbstractVertex}}, a) = Tuple(a), y -> a
Flux.functor(::Type{<:AbstractVertex}, v) = (base(v),), y -> v
Flux.functor(::Type{<:CompGraph}, g) = Tuple(vertices(g)), y -> g

"""
    fluxvertex(l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated())

Return a vertex wrapping the layer `l` with input vertex `in`.

Extra arguments `layerfun`, and `traitfun` can be used to add extra info about the vertex.
"""
fluxvertex(l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated()) = fluxvertex(layertype(l), l, in, layerfun, traitfun)

"""
    fluxvertex(name::String, l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated())

Return a vertex wrapping the layer `l` with input vertex `in` with name `name`.

Name is only used when displaying or logging and does not have to be unique (although it probably is a good idea).

Extra arguments `layerfun` and `traitfun` can be used to add extra info about the vertex.
"""
fluxvertex(name::String, l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated()) = fluxvertex(layertype(l), l, in, layerfun, traitfun ∘ named(name))

fluxvertex(::FluxParLayer, l, in::AbstractVertex, layerfun, traitfun) = absorbvertex(layerfun(MutableLayer(l)), in, traitdecoration = traitfun)

fluxvertex(::FluxDepthwiseConv, l, in::AbstractVertex, layerfun, traitfun) = absorbvertex(layerfun(MutableLayer(l)), in; traitdecoration=traitfun ∘ SizeNinNoutConnected)

fluxvertex(::FluxParInvLayer, l, in::AbstractVertex, layerfun, traitfun) = invariantvertex(layerfun(MutableLayer(l)), in, traitdecoration=traitfun)

fluxvertex(::FluxNoParLayer, l, in::AbstractVertex, layerfun, traitfun) = invariantvertex(layerfun(NoParams(l)), in, traitdecoration=traitfun)

# Decorate trait with extra stuff like logging of size changes or validation.
# Meant to be composable, e.g. using ∘
named(name) = t -> NamedTrait(t, name)
validated() = t -> SizeChangeValidation(t)
logged(;level = Base.CoreLogging.Debug, info = FullInfoStr()) = t -> SizeChangeLogger(level, info, t)

"""
   concat(v::AbstractVertex, vs::AbstractVertex...; traitfun=identity)

Return a vertex which concatenates input.

Inputs must have compatible activation shapes or an exception will be thrown.

Extra arguments `layerfun` and `traitfun` can be used to add extra info about the vertex.
"""
function concat(v::AbstractVertex, vs::AbstractVertex...; traitfun=identity, layerfun=identity)
    allactdims = unique(mapreduce(actdim, vcat, [v, vs...]))
    if length(allactdims) != 1
        throw(DimensionMismatch("Can not concatenate activations with different shapes! Got: $(join(allactdims, ", ", " and "))"))
    end

    allactranks = unique(mapreduce(actrank, vcat, [v, vs...]))
    if length(allactranks) != 1
        throw(DimensionMismatch("Can not concatenate activations with different shapes! Got:  $(join(allactranks, ", ", " and "))"))
    end

    conc(v, vs...; dims=allactdims[], traitdecoration=traitfun, outwrap=layerfun)
end

"""
    concat(name::String, v::AbstractVertex, vs::AbstractVertex...; traitfun=identity)

Return a vertex with name `name` which concatenates input.

Name is only used when displaying or logging and does not have to be unique (although it probably is a good idea).

Inputs must have compatible activation shapes or an exception will be thrown.

Extra arguments `layerfun` and `traitfun` can be used to add extra info about the vertex.
"""
concat(name::String, v::AbstractVertex, vs::AbstractVertex...; traitfun = identity, layerfun=identity) = concat(v, vs..., traitfun=traitfun ∘ named(name), layerfun=layerfun)

layer(v::AbstractVertex) = layer(base(v))
layer(v::CompVertex) = layer(v.computation)
layer(::InputVertex) = nothing

layertype(v::AbstractVertex) = layertype(base(v))
layertype(v::CompVertex) = layertype(v.computation)
layertype(::InputVertex) = nothing


actdim(v::AbstractVertex) = actdim.(layer.(NaiveNASlib.findterminating(v, inputs)))
actrank(v::AbstractVertex) = actrank.(layer.(NaiveNASlib.findterminating(v, inputs)))

mutate_weights(v::AbstractVertex, w) = mutate_weights(base(v), w)
mutate_weights(v::CompVertex, w) = mutate_weights(v.computation, w)

"""
    setlayer!(x, propval)

Set the properties `propval` to the layer wrapped in `x` where `propval` is a named tuple with fieldname->value pairs.

This typically means create a new layer with the given values and set the wrapped layer to it.

### Examples

```julia-repl
julia> v = fluxvertex(Dense(3, 4, relu), inputvertex("in", 3));

julia> layer(v)
Dense(3, 4, relu)   # 16 parameters

julia> NaiveNASflux.setlayer!(v, (;σ=tanh));

julia> layer(v)
Dense(3, 4, tanh)   # 16 parameters
```
"""
function setlayer!(x, propval) end
setlayer!(v::AbstractVertex, propval) = setlayer!(base(v), propval)
setlayer!(v::CompVertex, propval) = setlayer!(v.computation, propval)
setlayer!(m::AbstractMutableComp, propval) = setlayer!(wrapped(m), propval)
setlayer!(m::ResetLazyMutable, propval) = setlayer!(m.wrapped, propval)
setlayer!(m::MutationTriggered, propval) = setlayer!(m.wrapped, propval)
function setlayer!(m::MutableLayer, propval)
    m.layer = setproperties(m.layer, propval)
end