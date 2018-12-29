using Gtk

function preprocess(text)
    """
    preprocess
        this function call the model to process the data
        and return the predicted label

        :param text the input text for the model to process
        "return the label string to show
    """

    split_text = split(text)
    show_label = []

    for ele in split_text
        temp = [" " for i in range(1;length=min(5,length(ele)))]
        push!(show_label, *(temp...))
    end

    predict_label = ["O", "O", "O", "O"]

    for (index, ele) in enumerate(predict_label)
        temp = [show_label[index]...]
        temp[1:length(ele)] = [ele...]
        show_label[index] = join(temp, "")
    end

    return join(show_label, " ")
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
    GAccessor.text(sentence, text)
    label_text = preprocess(text)
    GAccessor.text(label, label_text)

end

signal_connect(on_button_click, b, "clicked")


set_gtk_property!(g, :column_homogeneous, true)
set_gtk_property!(g, :column_spacing, 15)
push!(win, g)
showall(win)
