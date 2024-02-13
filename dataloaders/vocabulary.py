def vocabulary(sentences, vocab_set):
    for sentence in sentences:
        for word in sentence:
            if word not in vocab_set:
                vocab_set.append(word)
    return vocab_set


def get_vocabulary(df): #_train, df_test): 
    eng_vocab_set = vocabulary(df['english'], ['<sos>', '<eos>', '<pad>', '<unk>'])
    spn_vocab_set = vocabulary(df['spanish'], ['<sos>', '<eos>', '<pad>', '<unk>'])

    return eng_vocab_set, spn_vocab_set


def vocab_index(df): #_train, df_test):
    eng_vocab, spn_vocab = get_vocabulary(df) #_train, df_test)
    print("Length English Vocabulary: ", len(eng_vocab))
    print("Length Spanish Vocabulary: ", len(spn_vocab))

    eng_vocab_index = dict([(word, i) for i, word in enumerate(eng_vocab)])
    spn_vocab_index = dict([(word, i) for i, word in enumerate(spn_vocab)])

    eng_index_vocab = dict([(i, word) for i, word in enumerate(eng_vocab)])
    spn_index_vocab = dict([(i, word) for i, word in enumerate(spn_vocab)])

    return eng_vocab_index, spn_vocab_index, eng_index_vocab, spn_index_vocab


def words_to_index(sentences, vocab_index):
    indexed_sentences = []
    for sent in sentences:
        sent_idx = [vocab_index[word] for word in sent]
        indexed_sentences.append(sent_idx)
    return indexed_sentences