"""
BERT-based Lexical Ambiguity Resolution.

Focuses on **lexical ambiguity** — words that have genuinely different meanings
(homonyms / polysemy) like "bank" (river bank vs financial institution),
"bat" (cricket bat vs flying mammal), "crane" (bird vs machine), etc.

Uses BERT contextual embeddings + WordNet glosses to pick the correct sense
and then generates a simple, clear sentence meaning.
"""

import torch
from transformers import BertTokenizer, BertModel

# ---------------------------------------------------------------------------
# Load model once at import time so subsequent calls are fast
# ---------------------------------------------------------------------------
_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
_model = BertModel.from_pretrained("bert-base-uncased")
_model.eval()


def _embed_text(text: str) -> torch.Tensor:
    """Return the mean-pooled embedding for *text*."""
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze(0)


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def disambiguate(word: str, sentence: str, synsets) -> dict:
    """
    Pick the best WordNet synset for *word* used in *sentence*.

    Returns dict with:
        - chosen_definition : str
        - chosen_synset_name : str
        - all_senses : list[dict]  (name, definition, score)
        - is_lexically_ambiguous : bool  (True if senses are genuinely different)
    """
    if not synsets:
        return {
            "chosen_definition": "No definitions found.",
            "chosen_synset_name": "",
            "all_senses": [],
            "is_lexically_ambiguous": False,
        }

    # Embed the full sentence for contextual representation
    sentence_emb = _embed_text(sentence)

    scored: list[dict] = []
    for syn in synsets:
        definition = syn.definition()
        # Build a gloss sentence: "word : definition"
        gloss = f"{word} : {definition}"
        gloss_emb = _embed_text(gloss)
        score = _cosine_similarity(sentence_emb, gloss_emb)
        scored.append({
            "name": syn.name(),
            "definition": definition,
            "score": round(score, 4),
        })

    # Sort descending by similarity
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Check if the word is truly lexically ambiguous:
    # The top two senses should be semantically different (low similarity)
    is_lexically_ambiguous = False
    if len(scored) >= 2:
        top_emb = _embed_text(f"{word} : {scored[0]['definition']}")
        second_emb = _embed_text(f"{word} : {scored[1]['definition']}")
        sense_similarity = _cosine_similarity(top_emb, second_emb)
        # If the top two senses are quite different, it's true lexical ambiguity
        is_lexically_ambiguous = sense_similarity < 0.92

    best = scored[0]
    return {
        "chosen_definition": best["definition"],
        "chosen_synset_name": best["name"],
        "all_senses": scored,
        "is_lexically_ambiguous": is_lexically_ambiguous,
    }


def generate_sentence_meaning(sentence: str, disambiguated_words: list[dict]) -> str:
    """
    Build a clear, understandable meaning of the whole *sentence*
    by replacing ambiguous words with the correct sense.

    Example:
        Input:  "I went to the bank to deposit money."
        Output: "I went to the financial institution to deposit money."
    """
    if not disambiguated_words:
        return "This sentence has a clear, straightforward meaning."

    # Build multiple candidate rephrasings and let BERT pick the best
    candidates = []

    # --- Candidate 1: Replace words with best synonym ---
    rephrased_syn = sentence
    for w in disambiguated_words:
        synonym = _get_best_synonym(w["word"], w.get("synset_name", ""))
        if synonym and synonym.lower() != w["word"].lower():
            rephrased_syn = _replace_word(rephrased_syn, w["word"], synonym)
    candidates.append(rephrased_syn)

    # --- Candidate 2: Replace words with short definition phrase ---
    rephrased_def = sentence
    for w in disambiguated_words:
        short_def = _short_definition(w["chosen_meaning"])
        rephrased_def = _replace_word(rephrased_def, w["word"], short_def)
    candidates.append(rephrased_def)

    # --- Candidate 3: "word means X" style, concise ---
    parts = []
    for w in disambiguated_words:
        short_def = _short_definition(w["chosen_meaning"])
        parts.append(f"'{w['word']}' here means {short_def}")
    clarification = "; ".join(parts)
    candidates.append(f"{sentence.rstrip('.')} — where {clarification}.")

    # Use BERT to pick the most natural candidate
    sentence_emb = _embed_text(sentence)
    best = candidates[0]
    best_score = -1.0

    for c in candidates:
        # Skip if the candidate is identical to the original sentence
        if c.strip().rstrip(".") == sentence.strip().rstrip("."):
            continue
        c_emb = _embed_text(c)
        score = _cosine_similarity(sentence_emb, c_emb)
        if score > best_score:
            best_score = score
            best = c

    # Clean up
    best = best.strip()
    if not best.endswith("."):
        best += "."
    return best


def _get_best_synonym(word: str, synset_name: str) -> str:
    """
    Get the clearest synonym for *word* from its chosen synset.
    Returns a clean, readable synonym or the word itself if none found.
    """
    from nltk.corpus import wordnet as wn

    if not synset_name:
        return word

    try:
        syn = wn.synset(synset_name)
    except Exception:
        return word

    # Get all lemma names from the synset
    lemma_names = [
        lemma.name().replace("_", " ")
        for lemma in syn.lemmas()
        if lemma.name().lower().replace("_", " ") != word.lower()
    ]

    if not lemma_names:
        return word

    # Pick the shortest, simplest synonym (most likely to be clear)
    # Prefer single-word synonyms, then shortest multi-word
    single_words = [l for l in lemma_names if " " not in l]
    if single_words:
        return min(single_words, key=len)
    return min(lemma_names, key=len)


def _short_definition(definition: str) -> str:
    """
    Shorten a WordNet definition to a clear 2-4 word phrase.
    E.g. "a financial institution that accepts deposits" → "financial institution"
    """
    definition = definition.strip()

    # Remove leading articles
    for prefix in ("a ", "an ", "the ", "to "):
        if definition.lower().startswith(prefix):
            definition = definition[len(prefix):]
            break

    # If already short, use as-is
    words = definition.split()
    if len(words) <= 3:
        return definition

    # Cut at common separators (that, which, ;, ,, or, etc.)
    import re
    # Split at "that", "which", ";", "," to get the core noun phrase
    core = re.split(r'\bthat\b|\bwhich\b|\bwhere\b|\bwhen\b|;|,', definition)[0].strip()
    core_words = core.split()

    if 1 <= len(core_words) <= 5:
        return core

    # Take up to 4 words
    return " ".join(core_words[:4])


def _replace_word(sentence: str, word: str, substitute: str) -> str:
    """Replace *word* in *sentence* (case-insensitive, first occurrence only)."""
    import re
    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
    return pattern.sub(substitute, sentence, count=1)