using Gtk
using Flux
using Flux: onehot
using Flux: onehotbatch
using Flux: crossentropy
using Flux: reset!
using Flux: throttle
using Flux: @show
using Flux.Optimise: SGD
using Printf

include("../loader.jl")
include("../lstm_custom.jl")
include("../predict_label.jl")
include("../Data/preprocess_helper.jl")

function upper_dim(EmbedDim, BatchSize)
    X -> reshape(X', (BatchSize, :, EmbedDim))
end

function construct_dict()
    """
    construct_dict
        this function is to construct the word and label dict from txt file
        modified based on loader.jl

        :return WordDict and LabelDict
    """

    WordDict = Dict()
    LabelDict = Dict()
    open("./demo/word_dict.txt") do f
        for l in eachline(f)
            Arr = split(l)
            WordDict[Arr[1]] = parse(Int, Arr[2])
        end
    end
    open("./demo/label_dict.txt") do f
        for l in eachline(f)
            Arr = split(l)
            LabelDict[Arr[2]] = Arr[1]
        end
    end
    return WordDict, LabelDict
end


function construct_model()
    """
    construct_model
        this function is used to reload the model for test
        modified based on the test part in main.jl
        :return model
    """

    model = load_cpu("best_model")
    Flux.testmode!(model)

    return model
end


function encode_text(text, WordDict)
    """
    encode_text
        this function gets a sentence as input,
        output the encoded sentence according to word_dict
        modified based on construct_test_data function in preprocess_helper,jl

        :param text the input sentence
        :param WordDict the trained word_dict
        :return a onehot matrix contained the encoded sentence [1, length of sentence, NumberofWords]
    """

    test_mat_x = []
    cur_len = 0
    cur_seq_x = []

    stop_word=[",", ":", ";", ".", ""]


    for word in split(text, " ")

        if(word in stop_word && cur_seq_x != [])
            push!(cur_seq_x, 1)
            cur_len += 1
            push!(test_mat_x, cur_seq_x)
            continue

        elseif word[1] == ""
                continue
        end

        word = lowercase(word)

        push!(cur_seq_x, get(WordDict, word, 3))
        cur_len += 1

    end

    return one_hot(reshape(Array(test_mat_x...), (1, cur_len)), length(WordDict), cur_len)

end


function get_label(text)
    """
    get_label
        this function call the model to process the data
        and return the predicted label

        :param text the input text for the model to process
        :return the label string to show
    """

    WordDict, LabelDict = construct_dict()
    ClassNum = length(LabelDict)
    split_text = encode_text(text, WordDict)
    model = construct_model()

    Output = upper_dim(ClassNum, 1)(model(split_text))
    Predict = predict_label(Output.data)

    split_sentence = split(text, " ")

    show_text = []
    show_label = []
    predict_labels = []

    for ele in split_sentence
        temp = [' ' for i in range(1;length=max(5,length(ele)))]
        push!(show_label, *(temp...))

        temp[1:length(ele)] = [ele...]
        push!(show_text, join(temp, ""))
    end

    for ele in Predict[1, :]
        push!(predict_labels, LabelDict[string(ele)])
    end


    for (index, ele) in enumerate(predict_labels)
        temp = [show_label[index]...]
        temp[1:length(ele)] = [ele...]
        show_label[index] = join(temp, "")
    end

    return join(show_text, " "), join(show_label, " ")
end



win = GtkWindow("A demo window")
g = GtkGrid()
text_input = GtkEntry()  # a widget for entering text
set_gtk_property!(text_input, :text, "Please enter your sentence")
b = GtkButton("Check")
sentence = GtkLabel("this region will show results")
label = GtkLabel("")

# Now let's place these graphical elements into the Grid:
g[1:3,1] = text_input    # Cartesian coordinates, g[x,y]
g[4,1] = b
g[1:4,2] = sentence  # spans both columns
g[1:4, 3] = label

function on_button_click(w)
    text = get_gtk_property(text_input, :text, String)

    text, label_text = get_label(text)
    GAccessor.text(label, label_text)
    GAccessor.text(sentence, text)
end

signal_connect(on_button_click, b, "clicked")


set_gtk_property!(g, :column_homogeneous, true)
set_gtk_property!(g, :column_spacing, 15)
push!(win, g)
showall(win)
