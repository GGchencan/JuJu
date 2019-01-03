include("loadembedding.jl")

embeddingfile = "ner.dim300.vec"
embeddim = 300
WordDict = Dict()
open("word_dict.txt") do f
    for l in eachline(f)
        Arr = split(l)
        WordDict[Arr[1]] = parse(Int, Arr[2])
    end
end

dense = load_embedding(embeddingfile, embeddim, WordDict)

print(dense(rand(length(WordDict))))
