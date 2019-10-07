
@testset "weight init" begin

    @testset "Warnings" begin
        @test_logs (:warn, r"Identity mapping not possible with nin != nout!") idmapping(2,3)
        @test_logs (:warn, r"Identity mapping not possible with nin != nout!") idmapping(1,1,3,4)
        @test_logs (:warn, r"Identity mapping requires odd kernel sizes!") idmapping(2,1,3,3)
        @test_logs (:warn, r"Identity mapping requires odd kernel sizes!") idmapping(1,2,3,3)
    end

    @testset "Dense ID mapping" begin
        l = Dense(3,3, initW = idmapping)
        indata = reshape(collect(Float32, 1:9), 3, 3)
        @test l(indata) == indata

        @test_logs (:warn, "Identity mapping not possible with nin != nout! Got nin=3, nout=4.") l = Dense(3,4, initW=idmapping)
        @test nin(l) == 3
        @test nout(l) == 4
    end

    @testset "Conv ID mapping" begin
        nch = 5
        l = Conv((3,5), nch=>nch, init=idmapping, pad=(2,1))
        indata = reshape(collect(Float32, 1:nch*7*13), 7, 13, nch, 1)
        @test l(indata) == indata

        @test_logs (:warn, "Identity mapping not possible with nin != nout! Got nin=$nch, nout=$(nch+2).") l = Conv((3,5), nch=> nch+2, init=idmapping, pad=(2,1))
        @test nin(l) == nch
        @test nout(l) == nch+2
    end

end
