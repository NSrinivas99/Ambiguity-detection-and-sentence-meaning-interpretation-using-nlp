"""
Paragraph-level **lexical ambiguity** analyser.

Given a paragraph, this module:
1. Splits text into sentences (spaCy).
2. For each sentence, identifies words with lexical ambiguity (multiple
   genuinely different meanings — homonyms / polysemy).
3. Uses BERT WSD (bert_model.py) to pick the correct sense in context.
4. Builds a simple, clear sentence-level meaning.
5. Returns a structured result for the frontend.
"""

import spacy
from nltk.corpus import wordnet as wn
from bert_model import disambiguate, generate_sentence_meaning

nlp = spacy.load("en_core_web_sm")

# Only consider these POS tags for ambiguity checks
_AMBIGUOUS_POS = {"NOUN", "VERB", "ADJ", "ADV"}

# Minimum number of WordNet synsets to consider a word potentially ambiguous
_MIN_SENSES = 2

# Common words that are technically polysemous but not interesting
_SKIP_WORDS = {
    "get", "go", "make", "take", "come", "give", "use", "find", "tell",
    "say", "know", "see", "want", "look", "think", "put", "try", "ask",
    "let", "call", "keep", "set", "run", "turn", "leave", "help", "start",
    "show", "hear", "play", "move", "live", "believe", "happen", "bring",
    "begin", "seem", "work", "need", "feel", "become", "thing", "way",
    "day", "time", "year", "people", "man", "woman", "child", "world",
    "life", "hand", "part", "place", "case", "point", "group", "number",
    "fact", "lot", "good", "great", "big", "long", "little", "old",
    "new", "first", "last", "own", "other", "right", "high", "small",
    "large", "next", "early", "young", "important", "few", "bad", "just",
    "able", "real", "sure", "different",
}


def _find_lexically_ambiguous_words(sent):
    """
    Return a list of dicts for every content word in *sent*
    that has genuine lexical ambiguity (truly different meanings).
    """
    results = []
    seen_lemmas = set()
    for token in sent:
        if (
            token.is_stop
            or token.is_punct
            or len(token.text) < 3
            or token.pos_ not in _AMBIGUOUS_POS
        ):
            continue

        lemma = token.lemma_.lower()
        if lemma in seen_lemmas or lemma in _SKIP_WORDS:
            continue
        seen_lemmas.add(lemma)

        synsets = wn.synsets(lemma)
        if len(synsets) >= _MIN_SENSES:
            # Check if the word has senses across different categories
            # (different POS or very different definitions = lexical ambiguity)
            pos_set = set(s.pos() for s in synsets)
            has_cross_pos = len(pos_set) > 1  # e.g. "bank" as noun & verb

            # Also check if there are multiple noun senses or verb senses
            # that are truly different
            has_multiple_distinct = len(synsets) >= 3

            if has_cross_pos or has_multiple_distinct:
                results.append({
                    "token": token,
                    "lemma": lemma,
                    "synsets": synsets,
                })
    return results


def interpret_sentence(text: str) -> dict:
    """
    Analyse *text* (a sentence or paragraph) and return a
    sentence-by-sentence interpretation focusing on lexical ambiguity.
    """
    doc = nlp(text)
    sentences = list(doc.sents)

    sentence_analyses: list[dict] = []

    for sent in sentences:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Find lexically ambiguous words in this sentence
        ambig_tokens = _find_lexically_ambiguous_words(sent)

        # Disambiguate each word and keep only truly ambiguous ones
        disambiguated: list[dict] = []
        for item in ambig_tokens:
            result = disambiguate(item["lemma"], sent_text, item["synsets"])

            # Only include words that are truly lexically ambiguous
            if result.get("is_lexically_ambiguous", True):
                disambiguated.append({
                    "word": item["token"].text,
                    "lemma": item["lemma"],
                    "pos": item["token"].pos_,
                    "chosen_meaning": result["chosen_definition"],
                    "synset_name": result["chosen_synset_name"],
                    "all_senses": [
                        {"definition": s["definition"], "score": s["score"]}
                        for s in result["all_senses"][:5]  # Top 5 senses only
                    ],
                })

        # Generate the overall sentence meaning
        sentence_meaning = generate_sentence_meaning(sent_text, disambiguated)

        sentence_analyses.append({
            "sentence": sent_text,
            "meaning": sentence_meaning,
            "ambiguous_words": disambiguated,
            "has_ambiguity": len(disambiguated) > 0,
        })

    # Build a human-readable summary
    n_sentences = len(sentence_analyses)
    n_ambiguous = sum(1 for s in sentence_analyses if s["has_ambiguity"])
    total_words = sum(len(s["ambiguous_words"]) for s in sentence_analyses)

    if total_words == 0:
        summary = "No lexical ambiguity was found in the text."
        meaning = "The text appears unambiguous — all words have clear meanings."
        explanation = "No words with multiple genuinely different meanings were detected."
    else:
        summary = (
            f"Found {total_words} lexically ambiguous word{'s' if total_words != 1 else ''} "
            f"across {n_ambiguous} sentence{'s' if n_ambiguous != 1 else ''}."
        )
        first = sentence_analyses[0]
        meaning = first["meaning"]
        explanation = (
            "Each sentence has been interpreted based on the correct contextual "
            "meaning of its lexically ambiguous words."
        )

    # Flatten ambiguous_words across all sentences
    all_ambiguous_words = []
    for sa in sentence_analyses:
        for w in sa["ambiguous_words"]:
            w["sentence"] = sa["sentence"]
            all_ambiguous_words.append(w)

    return {
        "sentence_analyses": sentence_analyses,
        "ambiguous_words": all_ambiguous_words,
        "summary": summary,
        "meaning": meaning,
        "explanation": explanation,
    }