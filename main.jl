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
include("loadembedding.jl")

"""
Get cleaned voc which counts at leat min_freq
add unk eos pad to dict
"""
EpochSize = 2
BatchSize = 10
EmbedSize = 300
HiddenSize = 300
Glove = false

TrainData, DevData, TestData, Dic, LabelDic = read_file()
DicSize = length(Dic)
ClassNum = length(LabelDic)

Test = one_epoch(TestData, BatchSize, DicSize, ClassNum)
Dev = one_epoch(DevData, BatchSize, DicSize, ClassNum)

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

if Glove
    EmbedLayer = load_embedding("./data/ner.dim300.vec", 300, Dic)
else
    EmbedLayer = Dense(DicSize, EmbedSize)
end

model = Chain(
    lower_dim(DicSize),
    EmbedLayer,
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


function eval_data(Data)
    count_ = 0
    tmpSum1 = 0
    tmpSum2 = 0
    tmpSum3 = 0
    Flux.testmode!(model)
    for d in Data
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
    P = tmpSum1 / tmpSum3
    R = tmpSum1 / tmpSum2
    F1 = 2 * P * R / (P + R)

    println("processed some tokens with $(tmpSum2) phrases; found: $(tmpSum3) phrases; correct: $(tmpSum1)")
    println( "precision: $(@sprintf("%.2f",P*100))%;  recall:  $(@sprintf("%.2f", R*100))%;  FB1:  $(@sprintf("%.2f",F1))")
    return (P, R, F1)
end


function train(EpochSize, ModelDir, Opt, loss)
    BestModel = "$(ModelDir)/best_model"
    BestPre = 0
    print("train begin.")
    for i = 1 : EpochSize
        Flux.testmode!(model, false)
        println("Epoch ", i)
        Data = one_epoch(TrainData, BatchSize, DicSize, ClassNum)
        # TotalBatch = sizeof(Data)
        BatchId = 0
        for D in Data
            Flux.train!(loss, [D], Opt)
            BatchId = BatchId + 1
            if BatchId % 10 == 0
                println("processed epoch $(i)/$(EpochSize), batch $(BatchId)")
            end
        end
        save_cpu(model, "$(ModelDir)/epoch-$(i)")
        Devp, Devr, Devf = eval_data(Dev)
        if Devp > BestPre
            BestPre = Devp
            save_cpu(model, BestModel)
        end
    end
    println(BestPre * 100)
end


# model = load_cpu("model")
# println(eval_data(Test))
# println(eval_data(Dev))

Lr = 0.1
Opt = SGD(params(model), Lr)

ModelDir = "./model_dir"
train(EpochSize, ModelDir, Opt, loss_with_mask)
