using Flux
function BiLSTM(data, EmbeddingSize, HiddenSize)
    """
    data(SeqLen,EmbeddingSize,BatchSize)
    Outputs(SeqLen, HiddenSizex2, BatchSize)
    """
    SeqLen = size(data, 1)
    BatchSize = size(data, 3)
    Forward = LSTM(EmbeddingSize, HiddenSize)
    Backward = LSTM(EmbeddingSize, HiddenSize)
    DataForward = [data[i,:,:] for i in 1:SeqLen]
    DataBackward = [data[i,:,:] for i in SeqLen:-1:1]
    ForwardOutput = Forward.(DataForward)
    BackwardOutput = Backward.(DataBackward)
    Outputs = [ForwardOutput[1];BackwardOutput[1]]
    for i in 2:SeqLen
        Outputs = cat(Outputs,[ForwardOutput[i];BackwardOutput[i]]; dims = 3)
    end
    Outputs = permutedims(Outputs, [3, 1,2])
    return Outputs
end

"""
Julia优先填充最后一个维度为目标，所以最小的单元放在第一个维度
example
SeqLen = 2
BatchSize = 5
EmbeddingSize = 2
HiddenSize = 3
data = rand(SeqLen,EmbeddingSize,BatchSize)
a = BiLSTM(data, EmbeddingSize, HiddenSize)
"""



