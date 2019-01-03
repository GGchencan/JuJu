using Flux
using Flux: onehot
using Flux: onehotbatch
using Flux: crossentropy
using Flux: reset!
using Flux: throttle
using Flux: @epochs
using Flux: @show
using Flux.Optimise: SGD
using Printf

include("loader.jl")
include("lstm_custom.jl")
include("predict_label.jl")
include("evaluate.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
EpochSize = 2
BatchSize = 10
EmbedSize = 50
HiddenSize = 300
ModelFn = "final_model.bson"

TrainData, TestData, Dic, LabelDic = read_file()
DicSize = length(Dic)
ClassNum = length(LabelDic)

println(DicSize)
println(ClassNum)
"""
Get train/test batch data
batch_size * seq_len * dic_size
"""
function lower_dim(Dim)
    X -> reshape(X, (:, Dim))'
end


function upper_dim(EmbedDim, BatchSize)
    X -> reshape(X', (BatchSize, :, EmbedDim))
end


function change_dim(Dim)
    X -> permutedims(X, Dim)
end

model = Chain(
    lower_dim(DicSize),
    Dense(DicSize, EmbedSize),
    upper_dim(EmbedSize, BatchSize),
    MyBiLSTM(EmbedSize, HiddenSize),
    Dropout(0.5),
    lower_dim(HiddenSize * 2),
    Dense(HiddenSize * 2, ClassNum),
    softmax
    )


function loss_with_mask(X, Y)
    LowerY = lower_dim(ClassNum)(Y)
    DimR = size(LowerY)[1]
    DimC = size(LowerY)[2]
    W = ones(DimR, DimC)
    W[DimR,:] = zeros(DimC)
    L = crossentropy(model(X), LowerY; weight = W)
    Flux.truncate!(model)
    @show(L)
    return L
end

Lr = 0.1
Opt = SGD(params(model), Lr)

Test = one_epoch(TestData, BatchSize, DicSize, ClassNum)

model = load_cpu("model")

for i = 1 : EpochSize
    println("epoch ", i)
    Data = one_epoch(TrainData, BatchSize, DicSize, ClassNum)
    for D in Data
        Flux.train!(loss_with_mask, [D], Opt)
    end
    save_cpu(model, "model-dropout")
end


count_ = 0
tmpSum1 = 0
tmpSum2 = 0
tmpSum3 = 0

for d in Test
    global count_
    global tmpSum1
    global tmpSum2
    global tmpSum3
    count_ = count_ + 1
    x = d[1]
    Output = upper_dim(ClassNum, BatchSize)(model(x))
    Predict = predict_label(Output.data)
    Truth = predict_label(d[2])
    acc = countChunks(Truth,Predict)
    tmpSum1 = tmpSum1 + acc[4]
    tmpSum2 = tmpSum2 + acc[5]
    tmpSum3 = tmpSum3 + acc[6]
end

println("Precision ",tmpSum1/tmpSum3)
println("Recall ",tmpSum1/tmpSum2)

println("tmpSum1 ", tmpSum1)
println("tmpSum2 ", tmpSum2)
println("tmpSum3 ", tmpSum3)
