using Flux
function load_embedding(embeddingfile, embdim, WordDict)
    word2vec = Dict()
    open(embeddingfile) do f
        for l in eachline(f)
            Arr = split(l)
            if size(Arr)[1] != embdim + 1
                println(l)
            else
                word2vec[Arr[1]] = Arr[2:end]
            end
        end
    end
    println("read ", length(word2vec), " word embeddings from ", embeddingfile)

    parsevector(arr) = map(x->parse(Float32,x), arr)
    word2vec = Dict(key => parsevector(word2vec[key]) for key in keys(word2vec))

    for vec in values(word2vec)
        if size(vec)[1] != embdim
            println("load error")
        end
    end
    wordnum = length(WordDict)
    weights = zeros(Float32,embdim, wordnum, )
    # defaultvec = rand(Float32,embdim)
    for key in keys(WordDict)
        v = get(word2vec,key, rand(Float32,embdim))
        column = WordDict[key]
        weights[:,column] = v
    end

    return Dense(weights, zeros(embdim), identity)
end
