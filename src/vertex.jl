

"""
    mutable(l, in::AbstractVertex, mutation=IoChange, traitfun=identity)

Return a mutable vertex wrapping the layer `l` with input vertex `in`.

Extra arguments `mutation` and `traitfun` can be used to change mutation type and to add extra info about the vertex.
"""
mutable(l, in::AbstractVertex, mutation=IoChange, traitfun=identity) = mutable(layertype(l), l, in, mutation, traitfun)

mutable(::FluxParLayer, l, in::AbstractVertex, mutation, traitfun) = absorbvertex(LazyMutable(MutableLayer(l)), nout(l), in, mutation=mutation, traitdecoration = traitfun)

mutable(::FluxParInvLayer, l, in::AbstractVertex, mutation, traitfun) =
invariantvertex(LazyMutable(MutableLayer(l)), in, mutation=mutation, traitdecoration=traitfun)

mutable(::FluxNoParLayer, l, in::AbstractVertex, mutation, traitfun) = invariantvertex(NoParams(l), in, mutation=mutation, traitdecoration=traitfun)

"""
   concat(v::AbstractVertex, vs::AbstractVertex...; mutation=IoChange, traitdecoration=identity)

Return a mutable vertex which concatenates input.

Inputs must have compatible activation shapes or an exception will be thrown.
"""
concat(v::MutationVertex, vs::MutationVertex...; mutation=IoChange, traitdecoration=identity) = concat(Val.(actdim.(layer.(vs))), mutation, traitdecoration, v, vs...)

concat(t, mutation, traitdecoration, v::AbstractVertex, vs::AbstractVertex...) = throw(ArgumentError("Can not concatenate activations with different shapes! Got: $t")) # I guess it might be doable, but CBA to try it out

concat(::Tuple{Val{N}}, mutation, traitdecoration, v::AbstractVertex, vs::AbstractVertex...) where N = conc(v, vs..., dims=N, mutation=mutation, traitdecoration=traitdecoration)


layer(v::AbstractVertex) = layer(base(v))
layer(v::CompVertex) = layer(v.computation)
