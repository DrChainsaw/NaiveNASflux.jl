@testset "Select" begin

    import NaiveNASflux: select

    @testset "Select parameters" begin
        mat = reshape(collect(1:3*4), 3, 4)

        @test select(mat, 2 => [2,-1,-1,4]) == [4 0 0 10;5 0 0 11;6 0 0 12]
        @test select(mat, 1 => [-1,1,3]) == [0 0 0 0;1 4 7 10;3 6 9 12]

        @test select(mat, 1 => [2,-1,3,-1], 2 => [-1,1,-1,4]) == [0 2 0 11;0 0 0 0;0 3 0 12;0 0 0 0]
    end

    @testset "KernelSizeAligned" begin
        pars = reshape(1:2*3*4*5, 5,4,3,2)
        ps = selectfilters(KernelSizeAligned(2, -1), pars)

        @test ps[1] == (1 => [-1, 1, 2, 3, 4, 5, -1])
        @test ps[2] == (2 => [2,3,4])
    end
end
