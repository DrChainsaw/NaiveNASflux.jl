
@testset "weight init" begin

    @testset "Dense ID mapping" begin
        @test_logs (:warn, r"Identity mapping not possible with nin != nout!") idmapping(2,3)
        l = Dense(3,3, initW = idmapping)
        indata = reshape(collect(Float32, 1:9), 3, 3)
        @test l(indata) == indata
    end

    @testset "Conv ID mapping" begin
        @test_logs (:warn, r"Identity mapping not possible with nin != nout!") idmapping(1,1,3,4)
        @test_logs (:warn, r"Identity mapping requires odd kernel sizes!") idmapping(2,1,3,3)
        @test_logs (:warn, r"Identity mapping requires odd kernel sizes!") idmapping(1,2,3,3)

        nch = 5
        l = Conv((3,5), nch=>nch, init=idmapping, pad=(2,1))
        indata = reshape(collect(Float32, 1:nch*7*13), 7, 13, nch, 1)
        @test l(indata) == indata
    end

end
