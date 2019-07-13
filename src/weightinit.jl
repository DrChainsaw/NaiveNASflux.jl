

function idmapping(nin,nout)
    nin == nout || @warn "Identity mapping not possible with nin != nout! Got nin=$nin, nout=$nout."
    return Matrix{Float32}(I, nin,nout)
end


function idmapping(w,h,nin,nout)
    nin == nout || @warn "Identity mapping not possible with nin != nout! Got nin=$nin, nout=$nout."
    w % 2 == 1 && h % 2 == 1 || @warn "Identity mapping requires odd kernel sizes! Got w=$w, h=$h."
    center_w = h ÷ 2 + 1
    center_h = w ÷ 2 + 1
    weights = zeros(Float32,h,w,nin,nout)
    for i in 1:min(nin,nout)
        weights[center_w, center_h, i, i] = 1
    end
    return weights
end
