

"""
    mutable(l, in::AbstractVertex, mutation=IoChange, traitfun=identity)

Return a mutable vertex wrapping the layer `l` with input vertex `in`.

Extra arguments `mutation` and `traitfun` can be used to change mutation type and to add extra info about the vertex.
"""
mutable(l, in::AbstractVertex, mutation=IoChange, traitfun=identity) = mutable(layertype(l), l, in, mutation, traitfun)

mutable(::FluxParLayer, l, in::AbstractVertex, mutation, traitfun) = absorbvertex(LazyMutable(MutableLayer(l)), nout(l), in, mutation=mutation, traitdecoration = traitfun)

mutable(::FluxParInvLayer, l, in::AbstractVertex, mutation, traitfun) =
invariantvertex(LazyMutable(MutableLayer(l)), in, mutation=mutation, traitdecoration=traitfun)

mutable(::FluxTransparentLayer, l, in::AbstractVertex, mutation, traitfun) = invariantvertex(NoParams(l), in, mutation=mutation, traitdecoration=traitfun)

"""
    conc(v::AbstractVertex, vs::AbstractVertex...; dims, mutation=IoChange, traitdecoration=identity)

Return a mutable vertex which concatenates input.

Inputs must have compatible activation shapes or an exception will be thrown.
"""
NaiveNASlib.conc(vs::AbstractVertex...; mutation=IoChange, traitdecoration=identity)
