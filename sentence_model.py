import spacy

nlp = spacy.load("en_core_web_sm")


def interpret_sentence(sentence):
    doc = nlp(sentence)

    verb = None
    obj = None
    instrument = None

    # find main verb
    for token in doc:
        if token.pos_ == "VERB" and verb is None:
            verb = token.text

    # find direct object
    for token in doc:
        if token.dep_ == "dobj":
            obj = token.text

    # find "with" phrase
    for token in doc:
        if token.text.lower() == "with":
            for child in token.children:
                if child.pos_ == "NOUN":
                    instrument = child.text

    # build final meaning
    if verb and obj and instrument:
        meaning = f"You used the {instrument} to {verb} the {obj}."
        explanation = f"The phrase 'with {instrument}' is interpreted as the instrument of action."

    elif verb and obj:
        meaning = f"Someone performed action '{verb}' on '{obj}'."
        explanation = "Basic sentence structure detected."

    else:
        meaning = "Could not determine exact meaning."
        explanation = "Sentence structure unclear."

    return {
        "meaning": meaning,
        "explanation": explanation
    }
