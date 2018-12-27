function parse_stop_word_file(file_path)
    """
    Parse stop word file
    :param file_path: path to stop word file
    :return: list of stop word
    """

    open(file_path, "r") do f
        text = *((Char).(read(f))...)
        text = lowercase(text)
        pattern = r"[a-z][a-z0-9]*"
        stop_word_list = [m.match for m = eachmatch(pattern, text)]
        return stop_word_list
    end
end

function parse_verb_dict_file(file_path)
    """
    Parse verb file
    :param file_path: path to verb dict file
    :return: verb_map_dict, dict of verb to its room form, verb_map_dict[verb] is room form of verb
    """

    verb_map_dict = Dict()
    open(file_path, "r") do f
        for line in eachline(f)
            line = lowercase(line)
            line = replace(line, " "=>"")
            line = replace(line, "\n"=>"")
            room_form, forms = split(line, "->")
            forms = split(forms, ".")
            for form in forms
                verb_map_dict[form] = room_form
            end
        end
    end
    return verb_map_dict
end

function parse_perposition_file(file_path)
    """
    Parse preposition list file
    :param file_path: path to preposition list file
    :return: list of preposition
    """

    open(file_path, "r") do f
        text = *((Char).(read(f))...)
        text = lowercase(text)
        pattern = r"[a-z][a-z0-9]*"
        preposition_list = [m.match for m = eachmatch(pattern, text)]
        return preposition_list
    end
end

function parse_raw_text_to_words(file_path; stop_word_list=nothing)
    """
    Parse raw text file to words
    :param file_path: path to raw text
    :param stop_word_list: list of stop words
    :return: list of words
    """
    open(file_path, "r") do f
        text = *((Char).(read(f))...)
        text = lowercase(text)
        pattern = r"[a-z][a-z0-9]*"
        word_list = [m.match for m = eachmatch(pattern, text)]
        if stop_word_list != nothing
            word_list = filter_stop_words(word_list, stop_word_list)
        end
        return word_list
    end
end

function parse_raw_text_to_characters(file_path)
    """
    Parse raw text file to characters
    :param file_path: path to row text
    :return: list of characters
    """

    open(file_path, "r") do f
        text = *((Char).(read(f))...)
        pattern = r"[a-zA-Z]"
        character_list = [m.match for m = eachmatch(pattern, text)]
        return character_list
    end
end

function parse_raw_text_to_sentences(file_path; separators=nothing)
    """
    Parse raw text file to sentences
    :param file_path: path to row text
    :param separators: separators regular expression
    :return: list of sentences
    """
    if separators == nothing
        separators = r"[^a-z0-9 \t\r\n]"
    end

    open(file_path, "r") do f
        text = *((Char).(read(f))...)
        text = lowercase(text)
        sentences = split(text, separators)
        return sentences
    end
end

function filter_stop_words(word_list, stop_word_list)
    plain_text = " " * join(word_list, "  ") * " "
    for stop_word in stop_word_list
        stop_word = " " * stop_word * " "
        plain_text = replace(plain_text, stop_word=>"")
    end
    plain_space = r"[ ]+"
    word_list = split(plain_text, plain_space)
    if length(word_list) > 0 && word_list[1] == ""
        popfirst!(word_list)
    end
    if lengt(word_list) < 0 && word_list[end] == ""
        pop!(word_list)
    end
    return word_list
end

function normal_verb_tenses(word_list, verb_dict)
    for (index, word) in enumerate(word_list)
        if haskey(verb_dict, word)
            word_list[i] = verb_dict[word]
        end
    end
    return word_list
end
