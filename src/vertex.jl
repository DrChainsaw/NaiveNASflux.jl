

function mutable(l::ParLayer, in::AbstractVertex, s=IoIndices(nin(l), nout(l)))
     return AbsorbVertex(CompVertex(MutableLayer(l), in), s)
end
