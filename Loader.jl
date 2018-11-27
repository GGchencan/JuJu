#using Flux: onehotbatch, onecold
using Random
using DelimitedFiles
##n*m -> n*m*class_num
function Onehot(array, class_num, idx)
    n=size(array)[1]
    m=size(array)[2]
    result=zeros(Float32, (n, idx, class_num))
    for i in 1:n
        for j in 1:idx
            result[i, j, array[i, j]]=1
        end
    end
    return result
end

#num_words means how many words in our dict, num_labels means how many labels
function Minibatch(data_dict, batch_size, num_words, num_labels)
    num_examples=size(data_dict['x'])[1]
    random_idx=Random.randperm(num_examples)[1:batch_size]


    mini_x=data_dict['x'][random_idx,:]
    row_sum=sum(mini_x, dims=1)
    n1=size(row_sum)[1]
    m1=size(row_sum)[2]
    idx=m1
    null=batch_size
    #print(row_sum)
    for i in 0:m1-1
        if(row_sum[n1, idx]!=null)
            break
        end
        idx-=1
    end
    #print(idx, '\n')

    return Onehot(data_dict['x'][random_idx,:], num_words, idx), Onehot(data_dict['y'][random_idx,:], num_labels, idx)
end

#function Readfile(training_file, testing_file, min_freq=10)
function Readfile()
    #special symbols
    #EOF:1
    #pad:2
    #less than min frequence:3

    training_dict=Dict() ##containg two arrays : 'x' contains featuers, 'y' contains labels
    testing_dict=Dict()
    ##please note the index in julia strating at 1, the same as matlab.
    ##a=[1, 2]. a[1]=1 and a[2]=2
    training_dict['x']=DelimitedFiles.readdlm("trn_x.txt", ' ', Int)
    training_dict['y']=DelimitedFiles.readdlm("trn_y.txt", ' ', Int)
    testing_dict['x']=DelimitedFiles.readdlm("test_x.txt", ' ', Int)
    testing_dict['y']=DelimitedFiles.readdlm("test_y.txt", ' ', Int)

    word_dict=Dict() ##map interger to word
    label_dict=Dict() ##map interger to label
    open("word_dict.txt") do f
        for l in eachline(f)
            arr=split(l)
            word_dict[arr[1]]=parse(Int, arr[2])
        end
    end
    open("label_dict.txt") do f
        for l in eachline(f)
            arr=split(l)
            label_dict[arr[1]]=parse(Int, arr[2])
        end
    end
    #print(word_dict["EOF"])
    return training_dict, testing_dict, word_dict, label_dict
end

batch_size=100

training_dict, testing_dict, word_dict, label_dict = Readfile()
mini_x, mini_y = Minibatch(training_dict, batch_size, length(word_dict), length(label_dict))
print("size of input x matrix: ", size(mini_x), " it has type", typeof(mini_x), '\n', "size of input y matrix: ", size(mini_y), " it has type", typeof(mini_y))
