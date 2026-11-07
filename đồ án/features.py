import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def token_documents(tokens, ngram_order):
    documents = []
    for sample in tokens:
        words = []
        steps, levels = sample.shape
        for level in range(levels):
            sequence = [f"l{level}c{int(sample[step, level])}" for step in range(steps)]
            words.extend(sequence)
            for order in range(2, ngram_order + 1):
                words.extend(f"l{level}n{order}_" + "_".join(sequence[start:start + order]) for start in range(steps - order + 1))
        words.extend("j_" + "_".join(f"c{int(code)}" for code in sample[step]) for step in range(steps))
        documents.append(" ".join(words))
    return documents


def fit_tfidf(train_tokens, test_tokens, ngram_order):
    vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+", sublinear_tf=True, norm="l2")
    train_matrix = vectorizer.fit_transform(token_documents(train_tokens, ngram_order))
    test_matrix = vectorizer.transform(token_documents(test_tokens, ngram_order))
    return vectorizer, train_matrix, test_matrix


def codebook_statistics(tokens, codebook_size):
    rows = []
    for level in range(tokens.shape[-1]):
        counts = np.bincount(tokens[:, :, level].reshape(-1), minlength=codebook_size).astype(np.float64)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -np.sum(probabilities * np.log(probabilities))
        rows.append({"level": level + 1, "used_codes": int(np.count_nonzero(counts)), "perplexity": float(np.exp(entropy))})
    return rows
