
"""
    InputShapeVertex <: AbstractVertex
    InputShapeVertex(v::AbstractVertex, t::FluxLayer)

Input type vertex which also has information about what type of layer the input is shaped for.
"""
struct InputShapeVertex{V<:AbstractVertex, L<:FluxLayer} <: AbstractVertex
    base::V
    t::L
end

@layer :expand InputShapeVertex

const inputshapemotivation = """
Providing the input type is not strictly necessary for the package to work and in many cases a normal `inputvertex` 
will do. 

One example of when it is useful is the [`concat`](@ref) function which needs to know the input type to
automatically determine which dimension to concatenate.
"""

"""
    inputvertex(name, size, type::FluxLayer)

Return an immutable input type vertex with the given `name` and `size` and a `type` which can be used to 
indicate what type of input is expected.

$inputshapemotivation
"""
NaiveNASlib.inputvertex(name, size, type::FluxLayer) = InputShapeVertex(inputvertex(name, size), type)

"""
    convinputvertex(name, nchannel, ndim) 

Return an input type vertex with the given `name` which promises convolution shaped input 
with `nchannel` channels and `ndim` number of dimensions for feature maps (e.g. 2 for images)
suitable for `Flux`s convolution layers.

$inputshapemotivation
"""
convinputvertex(name, nchannel, ndim) = inputvertex(name, nchannel, GenericFluxConvolutional{ndim}())

doc_convninputvertex(n) = """
    conv$(n)dinputvertex(name, nchannel)


Return an input type vertex with the given `name` which promises convolution shaped input 
with `nchannel` channels suitable for `Flux`s convolution layers.

Equivalent to [`convinputvertex(name, nchannel, $(n))`](@ref). 

$inputshapemotivation
"""

@doc doc_convninputvertex(1)
conv1dinputvertex(name, nchannel) = convinputvertex(name, nchannel, 1)
@doc doc_convninputvertex(2)
conv2dinputvertex(name, nchannel) = convinputvertex(name, nchannel, 2)
@doc doc_convninputvertex(3)
conv3dinputvertex(name, nchannel) = convinputvertex(name, nchannel, 3)


"""
    denseinputvertex(name, size)

Return an input type vertex with the given `name` which promises 2D shaped input
with `size` number of features suitable for e.g. `Flux`s `Dense` layer.

$inputshapemotivation
"""
denseinputvertex(name, size) = inputvertex(name, size, GenericFlux2D())

"""
    rnninputvertex(name, size)

Return an input type vertex with the given `name` which promises 2D shaped input
with `size` number of features suitable for `Flux`s recurrent layers.

$inputshapemotivation
"""
rnninputvertex(name, size) = inputvertex(name, size, GenericFluxRecurrent())

layertype(v::InputShapeVertex) = v.t
layer(v::InputShapeVertex) = LayerTypeWrapper(v.t) # so that layertype(layer(v)) works
NaiveNASlib.base(v::InputShapeVertex) = v.base
NaiveNASlib.name(v::InputShapeVertex) = name(base(v))
NaiveNASlib.nout(v::InputShapeVertex) = nout(base(v))
NaiveNASlib.nin(v::InputShapeVertex) = nin(base(v))
NaiveNASlib.outputs(v::InputShapeVertex) = outputs(base(v))
NaiveNASlib.inputs(::InputShapeVertex) = []

# Only to prevent stack overflow above
struct LayerTypeWrapper{L}
    t::L
end
layertype(l::LayerTypeWrapper) = l.t
Base.show(io::IO, l::LayerTypeWrapper) = show(io, l.t)

"""
    SizeNinNoutConnected <: NaiveNASlib.DecoratingTrait
    SizeNinNoutConnected(t)

Trait for computations for which a change in output size results in a change in input size but which 
is not fully `SizeTransparent`.

Example of this is grouped convolutions where output size must be an integer multiple of the input size.

Does not create any constraints or objectives, only signals that vertices after a 
`SizeNinNoutConnected` might need to change size if the size of the `SizeNinNoutConnected` vertex changes.
"""
struct SizeNinNoutConnected{T <: NaiveNASlib.MutationTrait} <: NaiveNASlib.DecoratingTrait
    base::T
end
NaiveNASlib.base(t::SizeNinNoutConnected) = t.base


NaiveNASlib.all_in_Δsize_graph(mode, ::SizeNinNoutConnected, args...) = NaiveNASlib.all_in_Δsize_graph(mode, SizeInvariant(), args...)

const doc_layerfun_and_traitfun = """
Keyword argument `layerfun` can be used to wrap the computation, e.g. in an [`ActivationContribution`](@ref). 

Keyword argument `traitfun` can be used to wrap the `MutationTrait` of the vertex in a `DecoratingTrait`
"""

"""
    fluxvertex(l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated())

Return a vertex which wraps the layer `l` and has input vertex `in`.

$doc_layerfun_and_traitfun
"""
fluxvertex(l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated()) = fluxvertex(layertype(l), l, in, layerfun, traitfun)

"""
    fluxvertex(name::AbstractString, l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated())

Return a vertex with name `name` which wraps the layer `l` and has input vertex `in`.

Name is only used when displaying or logging and does not have to be unique (although it probably is a good idea).

$doc_layerfun_and_traitfun
"""
fluxvertex(name::AbstractString, l, in::AbstractVertex; layerfun=LazyMutable, traitfun=validated()) = fluxvertex(layertype(l), l, in, layerfun, traitfun ∘ named(name))

fluxvertex(::FluxParLayer, l, in::AbstractVertex, layerfun, traitfun) = absorbvertex(layerfun(MutableLayer(l)), in, traitdecoration = traitfun)

fluxvertex(::FluxConvolutional, l, in::AbstractVertex, layerfun, traitfun) = absorbvertex(layerfun(MutableLayer(l)), in; traitdecoration= ngroups(l) == 1 ? traitfun : traitfun ∘ SizeNinNoutConnected)

fluxvertex(::FluxParInvLayer, l, in::AbstractVertex, layerfun, traitfun) = invariantvertex(layerfun(MutableLayer(l)), in, traitdecoration=traitfun ∘ FixedSizeTrait)

fluxvertex(::FluxNoParLayer, l, in::AbstractVertex, layerfun, traitfun) = invariantvertex(layerfun(NoParams(l)), in, traitdecoration=traitfun)


"""
    concat(v::AbstractVertex, vs::AbstractVertex...; traitfun=identity, layerfun=identity)

Return a vertex which concatenates input along the activation (e.g. channel if convolution, first dimension if dense) dimension.

Inputs must have compatible activation shapes or an exception will be thrown.

$doc_layerfun_and_traitfun

See also `NaiveNASlib.conc`. 
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
    concat(name::AbstractString, v::AbstractVertex, vs::AbstractVertex...; traitfun=identity, layerfun=identity)

Return a vertex with name `name` which concatenates input along the activation (e.g. channel if convolution, first dimension if dense) dimension.

Name is only used when displaying or logging and does not have to be unique (although it probably is a good idea).

Inputs must have compatible activation shapes or an exception will be thrown.

$doc_layerfun_and_traitfun

See also `NaiveNASlib.conc`. 
"""
concat(name::AbstractString, v::AbstractVertex, vs::AbstractVertex...; traitfun = identity, layerfun=identity) = concat(v, vs..., traitfun=traitfun ∘ named(name), layerfun=layerfun)

"""
    layer(v)

Return the computation wrapped inside `v` and inside any mutable wrappers.

# Examples
```jldoctest
julia> using NaiveNASflux, Flux

julia> layer(fluxvertex(Dense(2,3), inputvertex("in", 2)))
Dense(2 => 3)       # 9 parameters
```
"""
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
Dense(3 => 4, relu)  # 16 parameters

julia> NaiveNASflux.setlayer!(v, (;σ=tanh));

julia> layer(v)
Dense(3 => 4, tanh)  # 16 parameters
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