

mutable(l, in::AbstractVertex) = mutable(layertype(l), l, in)

function mutable(::FluxParLayer, l, in::AbstractVertex, s=IoIndices(nin(l), nout(l)))
     return AbsorbVertex(CompVertex(MutableLayer(l), in), s)
end
