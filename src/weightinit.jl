

function idmapping(nout,nin)
    nin == nout || @warn "Identity mapping not possible with nin != nout! Got nin=$nin, nout=$nout."
    return idmapping_nowarn(nout,nin)
end
idmapping_nowarn(nout, nin) = Matrix{Float32}(I, nout,nin)

function idmapping(h,w,nin,nout)
    nin == nout || @warn "Identity mapping not possible with nin != nout! Got nin=$nin, nout=$nout."
    w % 2 == 1 && h % 2 == 1 || @warn "Identity mapping requires odd kernel sizes! Got w=$w, h=$h."
    return idmapping_nowarn(h,w, nin,nout)
end
function idmapping_nowarn(h,w,nin,nout)
    center_w = h ÷ 2 + 1
    center_h = w ÷ 2 + 1
    weights = zeros(Float32,h,w,nin,nout)
    for i in 1:min(nin,nout)
        weights[center_w, center_h, i, i] = 1
    end
    return weights
end
