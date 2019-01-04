include("loadembedding.jl")

embeddingfile = "glove.6B\\glove.6B.300d.txt"
embeddim = 300
WordDict = Dict()
open("word_dict.txt") do f
    for l in eachline(f)
        Arr = split(l)
        WordDict[Arr[1]] = parse(Int, Arr[2])
    end
end

dense = load_embedding(embeddingfile, embeddim, WordDict)

print(dense(ones(length(WordDict))))

# the 0.04656 0.21318 -0.0074364 -0.45854 -0.035639 0.23643 -0.28836 0.21521 -0.13486 -1.6413 -0.26091 0.032434 0.056621 -0.043296
# WordDict["the"] = 13
print(dense.dense.W[:, 13])
# you will get the same vector like above
