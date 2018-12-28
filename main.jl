using Flux
using Flux: onehot
using Flux: onehotbatch
using Flux: crossentropy
using Flux: reset!
using Flux: throttle
using Flux: @epochs
using Flux: @show
using Flux.Optimise: SGD

include("loader.jl")
include("lstm_custom.jl")
include("predict_label.jl")
include("evaluate.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
EpochSize = 1
BatchSize = 1
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


function loss(X, Y)
    LowerY = lower_dim(ClassNum)(Y)
    L = crossentropy(model(X), LowerY)
    Flux.truncate!(model)
    @show(L)
    return L
end


function load_model(CheckPointFn)
    load_cpu(model, CheckPointFn)
end


Lr = 0.1
Opt = SGD(params(model), Lr)

test_sz = 1
Test = one_epoch(TestData, test_sz, DicSize, ClassNum)
Testd = Test(1)


for i = 1 : EpochSize
    println("epoch ", i)
    Data = one_epoch(TrainData, BatchSize, DicSize, ClassNum)
    for D in Data
        Flux.train!(loss, [D], Opt)
        """
        X = Testd[1]
        Output = upper_dim(ClassNum, test_sz)(model(X))
        Predict = predict_label(Output.data)
        Truth = predict_label(Testd[2])
        print("accuray is \n")
        print(countChunks(Truth, Predict))
        """
    end
    save_cpu(model, "model")
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
    print("predict is ", Predict)
    print("\n")
    Truth = predict_label(d[2])
    print("truth is ", Truth)
    print("\n")
    print("accuray is \n")
    acc = countChunks(Truth,Predict)
    tmpSum1 = tmpSum1 + acc[1]
    tmpSum2 = tmpSum2 + acc[2]
    tmpSum3 = tmpSum3 + acc[3]
    #tmp = vcat(tmp, acc)
    print(acc)
end

print("final1",tmpSum1/count_)
print("final2",tmpSum2/count_)
print("final3",tmpSum3/count_)

print("tmpSum1", tmpSum1)
print("tmpSum2", tmpSum2)
print("tmpSum3", tmpSum3)
