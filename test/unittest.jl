using Test
using Flux
using Flux.Tracker
include("../Dense_m.jl")
include("../lstm_custom.jl")

@testset "Dense_m test" begin
    word_num = 10
    seq_len = 7
    batch_size = 5
    embed_size = 2
    input_data = rand(batch_size, seq_len, word_num)
    weights = rand(embed_size, word_num)
    layer = Dense_m(weights)
    expected_output = zeros(batch_size, seq_len, embed_size)
    for i in 1:batch_size
        expected_output[i, :, :] = input_data[i, :, :] * (weights')
    end
    @test layer(input_data) == expected_output
end

@testset "MyBiLSTM test" begin
    seq_len = 10
    batch = 3
    in_dim = 20
    out_dim  = 30
    x = rand(batch,seq_len,  in_dim)
    y = ones(batch,seq_len,  2 * out_dim)
    bilstm = MyBiLSTM(in_dim, out_dim)
    ps = params(bilstm)
    loss(x, y) = Flux.mse(bilstm(x), y)
    l1 = loss(x, y )
    Tracker.back!(l1)
    grad1 =  ps[1].grad[1,1]
    ps[1].data[1,1] += 0.000001
    l2 = loss(x, y )
    grad2 = (l2 - l1) / 0.000001
    grad2 = Tracker.data(grad2)
    @test isapprox(abs(grad2), abs(grad1); atol=0.000001)
    # @test abs(grad2) ≈ abs(grad1)
end
