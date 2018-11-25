
#using Flux: onehotbatch, onecold
using Random

##n*m -> n*m*class_num
function Onehot(array, class_num)
    n=size(array)[1]
    m=size(array)[2]
    result=zeros(Float32, (n, m, class_num))
    for i in 1:n
        for j in 1:m
            result[i, j, array[i, j]]=1
        end
    end
    return result
end

#num_words means how many words in our dict, num_labels means how many labels
function Minibatch(data_dict, batch_size, num_words, num_labels)
    num_examples=size(data_dict['x'])[1]
    random_idx=Random.randperm(num_examples)[1:batch_size]
    return Onehot(data_dict['x'][random_idx,:], num_words), Onehot(data_dict['y'][random_idx,:], num_labels)
end

function Readfile(filename, min_freq=10)
    training_dict=Dict() ##containg two arrays : 'x' contains featuers, 'y' contains labels
    testing_dict=Dict()
    num_labels=10 ##how many labels
    num_words=3000 ##how many words in word dict

    num_training_sequences=1000  ##how many sequences in training data
    num_testing_sequences=1000 ##how many sequences in testing data
    max_seuqence_length=30 ##the length of longest sequence

    ##please note the index in julia strating at 1, the same as matlab.
    ##a=[1, 2]. a[1]=1 and a[2]=2
    training_dict['x']=rand(1:num_words, (num_training_sequences, max_seuqence_length))
    training_dict['y']=rand(1:num_labels, (num_training_sequences, max_seuqence_length))
    testing_dict['x']=rand(1:num_words, (num_testing_sequences, max_seuqence_length))
    training_dict['y']=rand(1:num_labels, (num_testing_sequences, max_seuqence_length))

    word_dict=Dict() ##map interger to word
    label_dict=Dict() ##map interger to label
    for i in 1:num_words
        word_dict[i]="word"
    end

    for i in 1:num_labels
        label_dict[i]="label"
    end

    #print(label_dict)
    return training_dict, testing_dict, word_dict, label_dict
end

batch_size=500

training_dict, testing_dict, word_dict, label_dict = Readfile("a")
mini_x, mini_y = Minibatch(training_dict, batch_size, length(word_dict), length(label_dict))
print("size of input x matrix: ", size(mini_x), " it has type", typeof(mini_x), '\n', "size of input y matrix: ", size(mini_y), " it has type", typeof(mini_y))
