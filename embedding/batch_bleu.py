import numpy as np
from embedding.bleu import compute_bleu


def batch_bleu(reference_batch, translation_batch, cumulative_bleu=False):
    """
    input formats are same with compute_blue. cumulative_blue is
     a flag for individual blue score calculation
    """
    if cumulative_bleu:
        bleu, precisions, bp, ratio, translation_length, reference_length = compute_bleu(reference_batch,
                                                                                         translation_batch)
    else:
        bleu_score = np.zeros(len(reference_batch))
        for i in range(len(reference_batch)):
            scores = compute_bleu([reference_batch[i]], [translation_batch[i]])
            bleu_score[i] = scores[0]
        bleu = np.mean(bleu_score)

    return round(bleu, 4)


if __name__ == '__main__':
    reference_corpus = [
        [
            ["woman", "in", "green", "shirt", "holding", "up", "a", "cellphone", "in", "her", "hand"],
            ["woman", "with", "green", "shirt", "talking", "on", "the", "cellphone"]
        ],
        [
            ["a", "person", "riding", "a", "bike", "with", "a", "tray", "of", "sandwiches", "on", "his", "head"],
            ["a", "man", "driving", "a", "car", "with", "sandwiches", "on", "his", "head"]
        ]
    ]

    translation_corpus = [
        ["woman", "with", "green", "shirt", "speaking", "holding", "the", "phone"],
        ["a", "man", "driving", "a", "car", "with", "sandwiches", "on", "his", "head"]
    ]

    print('BLEU score for all corpus :{}'.format(batch_bleu(reference_corpus,
                                                            translation_corpus,
                                                            cumulative_bleu=True)))

    print('BLEU score for individual '
          'sentences and mean of scores :{}'.format(batch_bleu(reference_corpus,
                                                              translation_corpus,
                                                              cumulative_bleu=False)))

