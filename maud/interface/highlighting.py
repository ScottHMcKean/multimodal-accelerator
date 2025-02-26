import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def simple_tokenize(text):
    """Simple word tokenizer that splits on whitespace and preserves punctuation"""
    return re.findall(r"\b\w+\b|[^\w\s]|\s+", text)


def stem_words(text):
    """Stem words using provided stemmer"""
    words = simple_tokenize(text.lower())
    return [stemmer.stem(word) for word in words if re.match(r"\b\w+\b", word)]


def highlight_stemmed_text(text, query_terms):
    """Highlight query terms in text using HTML mark tags, accounting for word stems."""
    # Get stems of the query terms
    stemmed_query_terms = set(stem_words(" ".join(query_terms)))

    # Split the text into words while preserving spaces and punctuation
    words = simple_tokenize(text)

    # Process each word and rebuild the text
    result = []
    for word in words:
        if re.match(r"\b\w+\b", word):  # If it's a word (not space/punctuation)
            stemmed_word = stemmer.stem(word.lower())
            if stemmed_word in stemmed_query_terms:
                result.append(f"<mark>{word}</mark>")
            else:
                result.append(word)
        else:
            result.append(word)  # Preserve spaces and punctuation

    return "".join(result)
