

mutable(l, in::AbstractVertex) = mutable(layertype(l), l, in)

function mutable(::FluxParLayer, l, in::AbstractVertex, s=IoIndices(nin(l), nout(l)))
     return AbsorbVertex(CompVertex(LazyMutable(MutableLayer(l)), in), s)
end

function mutable(::FluxParInvLayer, l, in::AbstractVertex, s=InvIndices(nin(l)))
     return InvariantVertex(CompVertex(LazyMutable(MutableLayer(l)), in), s)
end

function mutable(::FluxTransparentLayer, l, in::AbstractVertex)
     return InvariantVertex(CompVertex(l, in))
end
