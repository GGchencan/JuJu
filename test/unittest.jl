using Test

include("../Dense_m.jl")

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
    @test true
end
