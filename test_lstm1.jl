using Flux
include("lstm_custom.jl")
seq_len = 10
batch = 3
in_dim = 20
out_dim  = 30
x = ones(seq_len, batch, in_dim)
y = ones(seq_len, batch, 2 * out_dim)
bilstm = MyBiLSTM(in_dim, out_dim)
data = Iterators.repeated((x, y), 10)
loss(x, y) = Flux.mse(bilstm(x), y)
println(loss(x,y))
save_cpu(bilstm,"bilstm")
