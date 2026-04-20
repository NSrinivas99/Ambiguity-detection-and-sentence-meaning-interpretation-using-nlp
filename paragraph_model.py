import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")


def best_meaning(word, paragraph):
    senses = wn.synsets(word)

    if not senses:
        return word

    para_vec = nlp(paragraph)

    best = word
    best_score = -1

    for s in senses:
        def_vec = nlp(s.definition())
        score = para_vec.similarity(def_vec)

        if score > best_score:
            best_score = score
            best = s.lemma_names()[0]

    return best


def interpret_paragraph(paragraph):
    doc = nlp(paragraph)

    sentences = [sent.text for sent in doc.sents]

    interpreted_lines = []

    for sent in sentences:
        words = sent.split()
        new_sentence = []

        for w in words:
            senses = wn.synsets(w)

            if len(senses) > 1:
                new_w = best_meaning(w, paragraph)
                new_sentence.append(new_w)
            else:
                new_sentence.append(w)

        interpreted_lines.append(" ".join(new_sentence))

    # ensure output has 10 lines
    while len(interpreted_lines) < 10:
        interpreted_lines.append(" ")

    return interpreted_lines[:10]
