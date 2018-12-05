"""
Usage:
trueSeqs =    [1 7 2 3 8 10 10 10;2 6 9 2 3 10 10 10]
predictSeqs = [1 1 2 3 3 3 10 10 ;2 6 6 2 3 4 10 10]
f1 = countChunks(trueSeqs,predictSeqs)
println(f1)



example1:
input:
#tagString = ["2 B-ORG", "2 O", "3 B-MISC", "4 B-PER","5 I-PER","6 B-LOC","7 I-ORG","8 I-MISC","9 I-LOC"]
trueSeqs =    [1 7 2  3 8 10 10 10 10 10;2 6 9 2 3 1  7  10 10 10]
predictSeqs = [1 1 10 3 3 1  10 3   3 2 ;2 6 6 2 3 10 10 10 10 10]

ouput:
(0.14285714285714285, 0.2, 0.16666666666666666, 1.0, 5.0, 7.0)

example2:
input:
trueSeqs =    [10 10 10 10 10;2 10 10 10 10]
predictSeqs = [1 1 10 3 3 1  ;2 6 6 2 3 10]

output:
(0,0,0,0,0,0)

example3:
input:
trueSeqs =    [1 1 1 1 1 10 10 10 ;2 2 2 2 2 10 10 10]'
predictSeqs = [1 2 1 2 1 10 10 10 ;2 2 2 2 2 10 10 10 ]'
output:
(0.3333333333333333, 0.2, 0.25, 3.0, 5.0, 3.0)

input:
trueSeqs =    [1 1 1 1 1 ;2 2 2 2 2;1 7 2 3 8;2 6 9 2 3;1 7 2 6 2;2 2 2 2 3]'
predictSeqs = [1 2 1 2 1 ;2 2 2 2 2 ;1 1 2 3 3;2 6 6 2 3;1 2 3 4 4;5 7 8 9 3]'

output:
(0.2, 0.25, 0.22222222222222224, 3.0, 12.0, 15.0)

example4：
input
trueSeqs =    [1 1 10 10 10 ;2 10 10 10 10]
predictSeqs = [1 1 10 10 10 ;2 10 10 10 10 ]
output

(1.0, 1.0, 1.0, 2.0, 2.0, 2.0)

"""

function splitTag(chunkTag)
    if chunkTag == "O"
        return ("O","None")
    end
    return split(chunkTag,'-')
end

function startOfChunk( tag)
  chunkStart = (tag == "B")
return chunkStart
end

function endOfChunk(prevTag, tag)
  chunkEnd = (((prevTag == "B") & (tag == "B")) |((prevTag == "I") & (tag == "B"))
              | ((prevTag == "B") & (tag == "O")) |((prevTag == "I") & (tag == "O")))
  return chunkEnd
end


function countChunks(trueSeqs, predictSeqs)
    tagString = ["B-ORG", "O", "B-MISC", "B-PER","I-PER","B-LOC","I-ORG","I-MISC","I-LOC","O"]
    evaluate = zeros(3)
    startFlage= false
    prevPredPrefix = "O"
    prevTruePrefix = "O"
    prevTrueTag =  "O"
    prevPredTag  =  "O"
    N = size(trueSeqs,1)

    for i in 1:N
        M = findmax(trueSeqs[i,:])[2]
        for j in 1:M-1
            trueTag = tagString[trueSeqs[i,j]]
            predTag = tagString[predictSeqs[i,j]]
            truePrefix , trueType = splitTag(trueTag)
            PredPrefix , predType = splitTag(predTag)

            if startFlage ==  true
                trueEnd = endOfChunk(prevTruePrefix, truePrefix)
                predEnd = endOfChunk(prevPredPrefix, PredPrefix)
                if predEnd & trueEnd & (prevTrueTag== prevPredTag)
                    evaluate[1] += 1
                    startFlage= false
                elseif (predEnd != trueEnd) | (trueType != predType)
                    startFlage= false
                end
            end

            trueStart = startOfChunk(truePrefix)
            predStart = startOfChunk( PredPrefix)

            if trueStart & predStart & (trueType == predType)
                startFlage = true
            end
            if trueStart
                evaluate[2] += 1
            end
            if predStart
                evaluate[3] += 1
            end

            prevTruePrefix  = truePrefix
            prevPredPrefix  = PredPrefix
            prevTrueTag = trueTag
            prevPredTag  = predTag
        end
    end
    if startFlage == true
        evaluate[1] += 1
    end

    if (evaluate[1] == 0)
        return(0,0,0,0,0,0)
    end

    P = evaluate[1]/evaluate[3]
    R = evaluate[1]/evaluate[2]
    F_1 = 2*P*R / (P+R)

    return (P, R, F_1,evaluate[1],evaluate[2],evaluate[3])
end

"""
trueSeqs =    [1 7 2  3 8 10 10 10 10 10;2 6 9 2 3 1  7  10 10 10]
predictSeqs = [1 1 10 3 3 1  10 3   3 2 ;2 6 6 2 3 10 10 10 10 10]
f1 = countChunks(trueSeqs,predictSeqs)
println(f1)

trueSeqs =    [10 10 10 10 10;2 10 10 10 10]
predictSeqs = [1 1 10 3 3 1  ;2 6 6 2 3 10]
f1 = countChunks(trueSeqs,predictSeqs)
println(f1)
"""
