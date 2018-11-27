using Flux
function BiLSTM(data, Forward, Backward)
    '''
    normally BatchFirst is false,so we have
    data (SequenceLength,EmbeddingSize,BatchSize) Array{Float64,3}
    Forward and Backward are LSTM funcs (EmbedingSize, HiddenSize)
    output SequenceLength x (HiddenSizex2, BatchSize) Array{TrackedArray{…,Array{Float64,2}},1}
    '''
    SeqLen = size(data, 1)
    BatchSize = size(data, 3)
    DataForward = [data[i,:,:] for i in 1:SeqLen]
    DataBackward = [data[i,:,:] for i in SeqLen:-1:1]
    ForwardOutput = Forward.(DataForward)
    BackwardOutput = Backward.(DataBackward)
    HiddenSize = size(Forward.cell.Wh, 2)
    Outputs = [ [ForwardOutput[i];BackwardOutput[i]] for i in 1:SeqLen]
    return Outputs
    '''
    if you want 3 dimimension array (SequenceLength ,HiddenSizex2, BatchSize)
    use the code below
    Outputs = []
    for i in 1:SeqLen
        append!(Outputs, [[ForwardOutput[i];BackwardOutput[i]]])
    end
    still need to test
    '''
end

'''
example

'''
SeqLen = 9
BatchSize = 7
EmbeddingSize = 2
LSTM(2,3)
HiddenSize = 3
'''
#include('BiLSTM.jl')
data = rand(9,2,7)
Forward = LSTM(2,3)
Backward = LSTM(2,3)
BiLSTM(data, Forward, Backward)
9x(6,7)
'''
