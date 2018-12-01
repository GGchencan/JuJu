using Flux
include("lstm_custom.jl")
seq_len = 10
batch = 3
in_dim = 20
out_dim  = 30
data = rand(seq_len, batch, in_dim)
bilstm = MyBiLSTM(in_dim, out_dim)
out =  bilstm(data;batch_first=false)
print(size(out))
