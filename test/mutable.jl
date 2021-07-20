@testset "Mutable computation" begin

    import NaiveNASflux: AbstractMutableComp, MutableLayer, LazyMutable, weights, bias, select, mutate, hiddenweights, hiddenstate, state, outscale

    inszero = pairs((insert = (lt, pn) -> (args...) -> 0,))
    _nins(m) = [1:nin(m)[]]

    @testset "Dense MutableLayer" begin

        m = MutableLayer(Dense(2,3))

        @test nin(m) == [2]
        @test nout(m) == nout(m.layer) == 3
        @test m([1.0, 2.0]) == m.layer([1.0, 2.0])

        @test minΔninfactor(m) == 1
        @test minΔnoutfactor(m) == 1

        m.layer = Dense(3,4)
        bias(m.layer)[1:end] = 1:4
        @test nin(m) == [3]
        @test nout(m) == nout(m.layer) == 4
        @test m([1.0, 2.0, 3.0]) == m.layer([1.0, 2.0, 3.0])

        @testset "Select inputs" begin
            inds = [1,3]
            Wexp, bexp = weights(m.layer)[:, inds], bias(m.layer)
            NaiveNASlib.Δsize!(m, [inds], 1:nout(m))
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "Select outputs" begin
            inds = [1,2,4]
            Wexp, bexp = weights(m.layer)[inds, :], bias(m.layer)[inds]
            NaiveNASlib.Δsize!(m, _nins(m), inds)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "Insert inputs" begin
            inds = [1,-1, 2]
            Wexp, bexp = hcat(weights(m.layer)[:, 1], zeros(Float32, 3), weights(m.layer)[:, 2]), bias(m.layer)
            NaiveNASlib.Δsize!(m, [inds], 1:nout(m); inszero...)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "Insert outputs" begin
            inds = [-1, 1, -1, 3, -1]
            Wexp = permutedims(hcat(zeros(Float32, 3), weights(m.layer)[1, :], zeros(Float32, 3), weights(m.layer)[3, :], zeros(Float32,3)))
            bexp = Float32[0, bias(m.layer)[1], 0, bias(m.layer)[3], 0]
            NaiveNASlib.Δsize!(m, _nins(m), inds; inszero...)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "No bias" begin
            m = MutableLayer(Dense(rand(3,2), Flux.Zeros()))
            @test bias(layer(m)) == Flux.Zeros()

            @test nin(m) == [2]
            @test nout(m) == 3

            inds = [2,3]
            Wexp = weights(layer(m))[inds, :]
            NaiveNASlib.Δsize!(m,_nins(m), inds)
            assertlayer(layer(m), Wexp, Flux.Zeros())
        end
    end
    @testset "Convolutional layers" begin

        @testset "Conv MutableLayer" begin
            m = MutableLayer(Conv((2,3),(4=>5)))

            @test nin(m) == [4]
            @test nout(m) == nout(m.layer) == 5
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            input = reshape(collect(Float32, 1:3*4*4), 3, 4, 4, 1)
            @test m(input) == m.layer(input)

            @testset "Select inputs" begin
                inds = [1,3]
                Wexp, bexp = weights(m.layer)[:,:,inds,:], bias(m.layer)
                NaiveNASlib.Δsize!(m, [inds], 1:nout(m))
                assertlayer(m.layer, Wexp, bexp)
            end

            @testset "Select outputs" begin
                inds = [1,2,4]
                Wexp, bexp = weights(m.layer)[:,:,:,inds], bias(m.layer)[inds]
                NaiveNASlib.Δsize!(m, _nins(m), inds)
                assertlayer(m.layer, Wexp, bexp)
            end

            @testset "Insert inputs" begin
                inds = [1,-1, 2]
                wsize = deleteat!(collect(size(weights(m.layer))), 3)

                # Nothing beats working in four dimensions...
                # Stack 3D arrays in a 4:th dimension and then swap dim 3 and 4
                Wexp = permutedims(cat(
                weights(m.layer)[:,:,1,:],
                zeros(Float32, wsize...),
                weights(m.layer)[:,:,2,:], dims=4), [1,2,4,3])
                bexp = bias(m.layer)

                NaiveNASlib.Δsize!(m, [inds], 1:nout(m); inszero...)
                assertlayer(m.layer, Wexp, bexp)
            end

            @testset "Insert outputs" begin
                inds = [-1, 1, -1, 3, -1]
                wsize = deleteat!(collect(size(weights(m.layer))), 4)

                Wexp = cat(
                zeros(Float32, wsize...),
                weights(m.layer)[:,:,:,1],
                zeros(Float32, wsize...),
                weights(m.layer)[:,:,:,3],
                zeros(Float32,wsize...), dims=4)

                bexp = Float32[0, bias(m.layer)[1], 0, bias(m.layer)[3], 0]
                NaiveNASlib.Δsize!(m, _nins(m), inds; inszero...)
                assertlayer(m.layer, Wexp, bexp)
            end

            @testset "No bias" begin
                m = MutableLayer(Conv(Flux.convfilter((2,3), 4=>5), Flux.Zeros()))
                @test bias(layer(m)) == Flux.Zeros()

                @test nin(m) == [4]
                @test nout(m) == 5

                inds = [2,3]
                Wexp = weights(layer(m))[:,:,:,inds]
                NaiveNASlib.Δsize!(m, _nins(m), inds)
                assertlayer(layer(m), Wexp, Flux.Zeros())
            end
        end

        @testset "ConvTranspose MutableLayer" begin
            m = MutableLayer(ConvTranspose((2,3),(4=>5)))

            @test nin(m) == [4]
            @test nout(m) == nout(m.layer) == 5
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            input = reshape(collect(Float32, 1:3*4*4), 3, 4, 4, 1)
            @test m(input) == m.layer(input)

            inputs = [1,3]
            outputs = [1,2,4]
            Wexp, bexp = weights(m.layer)[:,:,outputs,inputs], bias(m.layer)[outputs]
            NaiveNASlib.Δsize!(m, [inputs], outputs)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "DepthwiseConv MutableLayer" begin
            m = MutableLayer(DepthwiseConv((2,2),(3=>6*3)))

            @test nin(m) == [3]
            @test nout(m) == nout(m.layer) == 18

            input = reshape(collect(Float32, 1:3*3*3), 3, 3, 3, 1)
            @test m(input) == m.layer(input)

            @testset "Select params" begin
                wins = [1, 3]
                wouts = [1, 2, 5, 6]
                outputs = mapreduce(i -> wouts .+ (i-1) .* 6, vcat, wins)
                Wexp, bexp = weights(m.layer)[:,:,wouts,wins], bias(m.layer)[outputs]
                NaiveNASlib.Δsize!(m, [wins], outputs)
                assertlayer(m.layer, Wexp, bexp)
                @test size(m(ones(Float32, 3,3,2,2)))[3:4] == (8, 2)
            end

            @testset "Insert params" begin
                inputs = [1, 2, -1]
                outputs = [1, 2, -1, -1, -1, -1, 3, 4, -1, -1, -1, -1]
                NaiveNASlib.Δsize!(m, [inputs], outputs)
            
                @test nin(m) == [3]
                @test nout(m) == 12

                @test size(m(ones(Float32, 3,3,3,2)))[3:4] == (12, 2)
            end
        end

        @testset "CrossCor MutableLayer" begin
            m = MutableLayer(CrossCor((2,3),(4=>5)))

            @test nin(m) == [4]
            @test nout(m) == nout(m.layer) == 5
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            input = reshape(collect(Float32, 1:3*4*4), 3, 4, 4, 1)
            @test m(input) == m.layer(input)

            inputs = [1,3]
            outputs = [1,2,4]
            Wexp, bexp = weights(m.layer)[:,:,inputs, outputs], bias(m.layer)[outputs]
            NaiveNASlib.Δsize!(m, [inputs], outputs)
            assertlayer(m.layer, Wexp, bexp)
        end
    end

    @testset "Diagonal MutableLayer" begin
        m = MutableLayer(Flux.Diagonal(4))

        @test nin(m) == [nout(m)] == [4]
        @test minΔninfactor(m) == 1
        @test minΔnoutfactor(m) == 1
        weights(m.layer)[1:end] = 1:4
        bias(m.layer)[1:end] = 1:4

        @test m(Float32[1,2,3,4]) == m.layer(Float32[1,2,3,4])

        @testset "Select params" begin
            inds = [1,3]
            Wexp, bexp = weights(m.layer)[inds], bias(m.layer)[inds]
            NaiveNASlib.Δsize!(m, [inds], inds)
            assertlayer(m.layer, Wexp, bexp)
        end

        @testset "Insert params" begin
            inds = [-1, 2, -1]
            Wexp, bexp = Float32[0, weights(m.layer)[2], 0], Float32[0, bias(m.layer)[2], 0]
            NaiveNASlib.Δsize!(m, [inds], inds; inszero...)
            assertlayer(m.layer, Wexp, bexp)
        end
    end

    @testset "Normalization layers" begin

        @testset "LayerNorm MutableLayer" begin
            m = MutableLayer(LayerNorm(3; affine=true))

            @test nin(m) == [nout(m)] == [3]
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1
            weights(m.layer.diag)[1:end] = 1:3
            bias(m.layer.diag)[1:end] = 1:3

            @test m(Float32[1,2,3]) == m.layer(Float32[1,2,3])

            @testset "Select params" begin
                inds = [1,3]
                Wexp, bexp = weights(m.layer.diag)[inds], bias(m.layer.diag)[inds]
                NaiveNASlib.Δsize!(m, [inds], inds)
                @test typeof(layer(m)) <: LayerNorm
                assertlayer(m.layer.diag, Wexp, bexp)
                @test nin(m) == [nout(m)] == [2]
            end

            @testset "Insert params" begin
                inds = [-1, 2, -1]
                Wexp, bexp = Float32[0, weights(m.layer.diag)[2], 0], Float32[0, bias(m.layer.diag)[2], 0]
                NaiveNASlib.Δsize!(m, [inds], inds; inszero...)
                @test typeof(layer(m)) <: LayerNorm
                assertlayer(m.layer.diag, Wexp, bexp)
                @test nin(m) == [nout(m)] == [3]
            end
        end

        function assertnorm(l, meanexp, varexp)
            @test l.β == meanexp
            @test l.γ == varexp
            @test vec(l.μ) == meanexp
            @test vec(l.σ²) == varexp
        end

        setpar(x) = x
        setpar(x::AbstractArray) = reshape(collect(Float32, 1:length(x)), size(x))

        @testset "$lab MutableLayer" for (l, lab) in (
                                (BatchNorm, BatchNorm),
                                (InstanceNorm, InstanceNorm),
                                ((n;kw...) -> GroupNorm(n,n; kw...), GroupNorm))

            m = MutableLayer(l(5; affine=true, track_stats=true))
            l_orig = layer(m)

            @test nin(m) == [nout(m)] == [5]
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            m.layer = Flux.fmap(setpar, layer(m))

            @testset "Select params" begin
                inds = [1,3,4]
                expall = Float32.(inds)
                NaiveNASlib.Δsize!(m, [inds], inds)
                assertnorm(m.layer, inds, inds)
                @test nin(m) == [nout(m)] == [3]
            end

            @testset "Insert params" begin
                NaiveNASlib.Δsize!(m, [[-1, 2, -1]], [-1, 2, -1])
                assertnorm(m.layer, [0, 3, 0], [1, 3, 1])
                @test nin(m) == [nout(m)] == [3]
            end
        end

        @testset "GroupNorm MutableLayer with groups" begin

            @testset "Groups of 2" begin
                m = MutableLayer(GroupNorm(6,3; affine=true, track_stats=true))
                m.layer = Flux.fmap(setpar, layer(m))
                inds = [1,2,5,6]
                NaiveNASlib.Δsize!(m, [inds], inds)
                @test layer(m).μ == [1, 3]
                @test layer(m).σ² == [1, 3]
            end

            @testset "Group size 8 to 9" begin
                # Now when dimensions don't add up: size 8 becomes size 9
                m = MutableLayer(GroupNorm(8,4; affine=true, track_stats=true))
                inds = [1,3,-1,-1,4,-1,7,-1,8]
                NaiveNASlib.Δsize!(m, [inds], inds)
                # Current alg for selecting which group to pick in this case is poor, don't wanna test it :)
                @test length(layer(m).μ) == 3
                @test length(layer(m).σ²) == 3
            end
        end
    end

    @testset "Recurrent layers" begin

        function assertrecurrent(l, Wiexp, Whexp, bexp, hexp, sexp)
            assertlayer(l, Wiexp, bexp)
            @test hiddenweights(l) == Whexp
            @test hiddenstate(l) == hexp
            @test state(l) == sexp
        end

        function setparsrnn(l)
            bias(l)[1:end] = collect(Float32, 1:nout(l)*outscale(l))
            hiddenstate(l)[1:end] = collect(Float32, 1:nout(l))
            state(l)[1:end] = collect(Float32, 1:nout(l))
        end

        function setparslstm(l)
            bias(l)[1:end] = collect(Float32, 1:nout(l)*outscale(l))
            foreach(h -> h[1:end] = collect(Float32, 1:nout(l)), hiddenstate(l))
            foreach(h -> h[1:end] = collect(Float32, 1:nout(l)), state(l))
        end

        @testset "RNN MutableLayer" begin
            m = MutableLayer(RNN(3, 4))
            setparsrnn(layer(m))

            @test nin(m) == [3]
            @test nout(m) == nout(m.layer) == 4
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            @testset "Select inputs" begin
                inds = [1, 3]
                Wiexp = weights(layer(m))[:, inds]
                Whexp = copy(hiddenweights(layer(m)))
                bexp = copy(bias(layer(m)))
                hexp = copy(hiddenstate(layer(m)))
                sexp = copy(state(layer(m)))
                
                NaiveNASlib.Δsize!(m, [inds], 1:nout(m))
                assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)
            end

            @testset "Insert outputs" begin
                inds = [1,-1, 2]
                Wiexp = permutedims(hcat(weights(layer(m))[1, :], zeros(Float32, 2), weights(layer(m))[2, :]))
                wh = hiddenweights(layer(m))
                Whexp = [wh[1, 1] 0 wh[1, 2]; zeros(Float32, 1, 3); wh[2, 1] 0 wh[2, 2]]
                bexp = Float32[bias(layer(m))[1], 0, bias(layer(m))[2]]
                hexp = Float32[hiddenstate(layer(m))[1], 0, hiddenstate(layer(m))[2]] |> hcat
                sexp = Float32[state(layer(m))[1], 0, state(layer(m))[2]] |> hcat
                NaiveNASlib.Δsize!(m, _nins(m), inds; inszero...)
                assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)
            end

            #Sanity check that the layer still seems to work after mutation
            output = m(reshape(collect(Float32, 1:2*10), 2,10))
            @test size(output) == (3, 10)
            @test isnan.(output) == falses(size(output))
        end

        @testset "LSTM MutableLayer" begin
            m = MutableLayer(LSTM(3, 4))
            setparslstm(layer(m))

            @test nin(m) == [3]
            @test nout(m) == nout(m.layer) == 4
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            @testset "Select inputs" begin
                inds = [1, 3]
                Wiexp = weights(layer(m))[:, inds]
                Whexp = copy(hiddenweights(layer(m)))
                bexp = copy(bias(layer(m)))
                hexp = copy(hiddenstate(layer(m)))
                sexp = copy(state(layer(m)))
                NaiveNASlib.Δsize!(m, [inds], 1:nout(m))
                assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)
            end

            @testset "Insert outputs" begin
                inds = [1,-1, 2]
                wi = weights(layer(m))
                scalerange = (0:outscale(layer(m))-1) .* nout(layer(m))
                Wiexp = permutedims(mapfoldl(offs -> hcat(wi[1+offs, :], zeros(Float32, 2), wi[2+offs, :]), hcat, scalerange))
                wh = hiddenweights(layer(m))
                Whexp = mapfoldl(offs -> [wh[1+offs, 1] 0 wh[1+offs, 2]; zeros(Float32, 1, 3); wh[2+offs, 1] 0 wh[2+offs, 2]], vcat, scalerange)
                bexp = mapfoldl(offs -> Float32[bias(layer(m))[1+offs], 0, bias(layer(m))[2+offs]], vcat, scalerange)
                hexp = map(hs -> Float32[hs[1], 0, hs[2]] |> hcat, hiddenstate(layer(m)))
                sexp = map(hs -> Float32[hs[1], 0, hs[2]] |> hcat, state(layer(m)))
                NaiveNASlib.Δsize!(m, _nins(m), inds; inszero...)
                assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)
            end

            #Sanity check that the layer still seems to work after mutation
            output = m(reshape(collect(Float32, 1:2*10), 2,10))
            @test size(output) == (3, 10)
            @test isnan.(output) == falses(size(output))
        end

        @testset "GRU MutableLayer" begin
            m = MutableLayer(GRU(3, 4))
            setparsrnn(layer(m))

            @test nin(m) == [3]
            @test nout(m) == nout(m.layer) == 4
            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            @testset "Select inputs" begin
                inds = [1, 3]
                Wiexp = weights(layer(m))[:, inds]
                Whexp = copy(hiddenweights(layer(m)))
                bexp = copy(bias(layer(m)))
                hexp = copy(hiddenstate(layer(m)))
                sexp = copy(state(layer(m)))
                NaiveNASlib.Δsize!(m, [inds], 1:nout(m))
                assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)
            end

            @testset "Insert outputs" begin
                inds = [1,-1, 2]
                wi = weights(layer(m))
                scalerange = (0:outscale(layer(m))-1) .* nout(layer(m))
                Wiexp = permutedims(mapfoldl(offs -> hcat(wi[1+offs, :], zeros(Float32, 2), wi[2+offs, :]), hcat, scalerange))
                wh = hiddenweights(layer(m))
                Whexp = mapfoldl(offs -> [wh[1+offs, 1] 0 wh[1+offs, 2]; zeros(Float32, 1, 3); wh[2+offs, 1] 0 wh[2+offs, 2]], vcat, scalerange)
                bexp = mapfoldl(offs -> Float32[bias(layer(m))[1+offs], 0, bias(layer(m))[2+offs]], vcat, scalerange)
                hexp = Float32[hiddenstate(layer(m))[1], 0, hiddenstate(layer(m))[2]] |> hcat
                sexp = Float32[state(layer(m))[1], 0, state(layer(m))[2]] |> hcat
                NaiveNASlib.Δsize!(m, _nins(m), inds; inszero...)
                assertrecurrent(layer(m), Wiexp, Whexp, bexp, hexp, sexp)
            end

            #Sanity check that the layer still seems to work after mutation
            output = m(reshape(collect(Float32, 1:2*10), 2,10))
            @test size(output) == (3, 10)
            @test isnan.(output) == falses(size(output))
        end
    end

    @testset "Clone MutableLayer" begin
            m = MutableLayer(Dense(2,3))
            cloned = clone(m)
            @test layer(cloned) !== layer(m)
            @test cloned([1, 2]) == m([1, 2])
    end

    @testset "LazyMutable" begin

        @testset "LazyMutable with Dense MutableLayer" begin
            m = MutableLayer(Dense(3,4))
            mlazy = LazyMutable(m)

            Wexp = weights(layer(m))
            bexp = bias(layer(m))

            @test minΔninfactor(m) == 1
            @test minΔnoutfactor(m) == 1

            NaiveNASlib.Δsize!(mlazy, [[1, 3]], 1:nout(m))
            assertlayer(layer(m), Wexp, bexp)

            NaiveNASlib.Δsize!(mlazy, _nins(mlazy), [2, 4, -1]; inszero...)
            assertlayer(layer(m), Wexp, bexp)

            @test layer(m) == layer(mlazy)
            @test layertype(m) == layertype(mlazy)

            expected = mlazy(Float32[2,3])

            @test nin(mlazy) == nin(m) == [2]
            @test nout(mlazy) == nout(m) == 3

            @test expected == m(Float32[2,3])
        end

        @testset "LazyMutable DepthwiseConv" begin
            m = LazyMutable(MutableLayer(DepthwiseConv((2,2),(3=>6*3))))

            @test nin(m) == [3]
            @test nout(m) == 18

            input = reshape(collect(Float32, 1:3*3*3), 3, 3, 3, 1)
            @test m(input) == layer(m)(input)

            wins = [1, 3]
            wouts = [1, 2, 5, 6]
            outs = mapreduce(i -> wouts .+ (i-1) .* 6, vcat, wins)
            Wexp, bexp = weights(layer(m))[:,:,wouts,wins], bias(layer(m))[outs]

            NaiveNASlib.Δsize!(m, [wins], outs)
            @test size(m(ones(Float32, 3,3,2,2)))[3:4] == (8, 2)

            assertlayer(layer(m), Wexp, bexp)
        end

        @testset "LazyMutable reselect" begin
            m = LazyMutable(MutableLayer(Dense(5,5)))

            NaiveNASlib.Δsize!(m, [[-1, 1, 3, -1, 4]], 1:nout(m))
            @test m.inputs == [[-1, 1, 3, -1, 4]]

            NaiveNASlib.Δsize!(m, [[1,2,4,5]], 1:nout(m))
            @test m.inputs == [[-1, 1,-1, 4]]

            NaiveNASlib.Δsize!(m, _nins(m), [2, -1, 3, -1, 4])
            @test m.outputs == [2, -1, 3, -1, 4]
            @test m.inputs == [[-1, 1,-1, 4]]

            NaiveNASlib.Δsize!(m, _nins(m), [1, 2, 5,-1])
            @test m.outputs == [2, -1, 4, -1]
            @test m.inputs == [[-1, 1,-1, 4]]

            @test m(Float32[1,3,5,7]) == layer(m)(Float32[1,3,5,7])
        end

        @testset "Clone" begin
            mlazy = LazyMutable(MutableLayer(Dense(2,3)))
            cloned = clone(mlazy)
            @test layer(cloned) !== layer(mlazy)
            @test cloned([1, 2]) == mlazy([1, 2])
        end

        @testset "Functor" begin
            m = LazyMutable(MutableLayer(Dense(2,3)))
            visitfun(x) = x
            visitdense = false
            function visitfun(l::AbstractArray{Float32})
                visitdense = true
                return l
            end

            NaiveNASlib.Δsize!(m, [[-1, -1, 1, 2]], 1:nout(m))
            @test size(weights(layer(m))) == (3,2)

            Flux.fmap(visitfun, m)
            @test visitdense

            @test size(weights(layer(m))) == (3,4)
        end

        @testset "Force mutation" begin
            import NaiveNASflux: FluxDense
            invertex = inputvertex("in", 3, FluxDense())
            hlayer = fluxvertex("hlayer", Dense(3,4), invertex)
            outlayer = fluxvertex("outlayer", Dense(4, 2), hlayer)
            graph = CompGraph(invertex, outlayer)

            Δnout!(hlayer, 2)

            @test nout(layertype(hlayer), layer(hlayer)) == 4
            @test nin(layertype(hlayer), layer(outlayer)) == 4

            NaiveNASflux.forcemutation(graph)

            @test nout(layertype(hlayer), layer(hlayer)) == 6
            @test nin(layertype(hlayer), layer(outlayer)) == 6
        end
    end
end
