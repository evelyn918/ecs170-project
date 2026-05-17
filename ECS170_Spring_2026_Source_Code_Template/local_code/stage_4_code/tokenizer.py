import os

def create_dataset(datapath):
    print("Creating dataset...")
    labels = []
    full_review = []
    print("Reading negative reviews...")
    with os.scandir((datapath + "/neg")) as folder:
        for file in folder:
            sentence = []
            with open(file.path, 'r', encoding='utf-8') as text_file:
                for line in text_file:
                    for word in line.split():
                        token_word, symbols = tokenize(word)

                        sentence.append(token_word)

                        if symbols:
                            # This is just to make sure "" gets stored instead of ","
                            i = 0
                            while i < len(symbols) - 1:
                                if symbols[i] == '""' and (i + 1) <= (len(symbols) - 1) and symbols[i + 1] == '"':
                                    sentence.append('""')
                                    #print("Quotes found")
                                    i += 2
                                else:
                                    sentence.append(symbols[i])
                                    i += 1
            #print(sentence)
            # filter out br and other things like < and >
            # also could use some recursion to better split that's into that and 's
            full_review.append(sentence)
            labels.append(0)

    print("Reading positive reviews...")
    with os.scandir((datapath + "/pos")) as folder:
        for file in folder:
            sentence = []
            with open(file.path, 'r', encoding='utf-8') as text_file:
                for line in text_file:
                    for word in line.split():
                        token_word, symbols = tokenize(word)

                        sentence.append(token_word)

                        if symbols:
                            # This is just to make sure "" gets stored instead of ","
                            i = 0
                            while i < len(symbols) - 1:
                                if symbols[i] == '"' and (i + 1) <= (len(symbols) - 1) and symbols[i + 1] == '"':
                                    sentence.append('""')
                                    #print("Quotes found")
                                    i += 2
                                else:
                                    sentence.append(symbols[i])
                                    i += 1
            #print(sentence)
            full_review.append(sentence)
            labels.append(1)

    #counter = 0
    #with open("negative.txt", 'w', encoding='utf-8') as file:
    #    for text in full_review:
    #        file.write(" ".join(text) + str(labels[counter]) + '\n')
    #        counter += 1

    sections = ["text", "label"]
    print("Dataset created...")
    return dict(zip(sections, [full_review, labels]))


# standardizes words and separate symbols from words. ex (This)? returns [this]  and [()?]
def tokenize(word):
    token = []
    special_symbols = []

    for letter in word:
        if letter.isalpha() or letter == "'":
            if letter.isupper():
                letter = letter.lower()
                token.append(str(letter))
            else:
                token.append(letter)
        else:
            special_symbols.append(letter)
    token = "".join(token)

    return token, special_symbols