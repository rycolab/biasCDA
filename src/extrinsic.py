import pickle
from tqdm import tqdm
import numpy as np


def save_stereotypes(animate_file, text_file, out_file):
    """
    Save list of words that are stereotyped towards men or women
    :param animate_file: list of noun pairs
    :param text_file: file to test words counts on
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    words = list(zip([line[1] for line in lines], [line[2] for line in lines]))
    with open(text_file) as f:
        text = f.read()
    text = text.split()
    fem_main = []
    masc_main = []
    for i in tqdm(range(len(words)), total=len(words)):
        fem, masc = words[i]
        fem_count = text.count(fem) + text.count(fem.capitalize())
        masc_count = text.count(masc) + text.count(masc.capitalize())
        if .25 * fem_count >= masc_count and fem_count != 0:
            fem_main.append((i, fem, masc))
        elif .25 * masc_count >= fem_count and masc_count != 0:
            masc_main.append((i, fem, masc))
    print(len(fem_main), len(masc_main))
    with open(out_file, "wb") as f:
        pickle.dump(fem_main, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(masc_main, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_probs(prob_file):
    """
    :param prob_file: File containing query probabilities
    :return: list of negative log likelihoods
    """
    with open(prob_file, "r") as f:
        lines = f.readlines()
    probs = [float(line.strip()) for line in lines]
    return probs


def calc_romance_bias(probs):
    """
    :param probs: list of negative log likelihoods for a romance language corpus
    :return: gender bias in corpus
    """
    bias = 0
    for idx in range(0, len(probs), 32):
        bias -= probs[idx + 1] + probs[idx + 5] + probs[idx + 9] + probs[idx + 13]
        bias += probs[idx + 18] + probs[idx + 22] + probs[idx + 26] + probs[idx + 30]
    return bias / 8


def calc_romance_grammar(probs):
    """
    :param probs: list of negative log likelihoods for a romance language corpus
    :return: grammaticality of corpus
    """
    grammar = 0
    for idx in range(0, len(probs), 32):
        grammar -= probs[idx + 1] + probs[idx + 5] + probs[idx + 9] + probs[idx + 13]
        grammar -= probs[idx + 18] + probs[idx + 22] + probs[idx + 26] + probs[idx + 30]
        grammar += probs[idx] + probs[idx + 4] + probs[idx + 8] + probs[idx + 12]
        grammar += probs[idx + 19] + probs[idx + 23] + probs[idx + 27] + probs[idx + 31]
    return grammar / 4


def calc_hebrew_bias(probs):
    """
    :param probs: list of negative log likelihoods for a Hebrew corpus
    :return: gender bias in corpus
    """
    bias = 0
    for idx in range(0, len(probs), 16):
        bias -= probs[idx + 1] + probs[idx + 5] + probs[idx + 9] + probs[idx + 13]
        bias += probs[idx + 2] + probs[idx + 6] + probs[idx + 10] + probs[idx + 14]
    return bias / 4


def calc_hebrew_grammar(probs):
    """
    :param probs: list of negative log likelihoods for a Hebrew corpus
    :return: grammaticality of corpus
    """
    grammar = 0
    for idx in range(0, len(probs), 16):
        grammar -= probs[idx + 1] + probs[idx + 5] + probs[idx + 9] + probs[idx + 13]
        grammar -= probs[idx + 2] + probs[idx + 6] + probs[idx + 10] + probs[idx + 14]
        grammar += probs[idx] + probs[idx + 4] + probs[idx + 8] + probs[idx + 12]
        grammar += probs[idx + 19] + probs[idx + 23] + probs[idx + 27] + probs[idx + 31]
    return grammar / 2


def calc_russian_bias(probs):
    """
    :param probs: list of negative log likelihoods for a Russian coprus
    :return: gender bias in corpus
    """
    bias = 0
    for idx in range(0, len(probs), 24):
        bias -= probs[idx + 1] + probs[idx + 3] + probs[idx + 5] + probs[idx + 7]
        bias += probs[idx + 8] + probs[idx + 10] + probs[idx + 12] + probs[idx + 14]
        bias -= probs[idx + 17] + probs[idx + 19] + probs[idx + 21] + probs[idx + 23]
        bias += probs[idx + 16] + probs[idx + 18] + probs[idx + 20] + probs[idx + 22]
    return bias / 4


def calc_russian_grammar(probs):
    """
    :param probs: list of negative log likelihoods for a Russian corpus
    :return: grammaticality of corpus
    """
    grammar = 0
    for idx in range(0, len(probs), 16):
        grammar -= probs[idx + 1] + probs[idx + 5] + probs[idx + 9] + probs[idx + 13]
        grammar -= probs[idx + 2] + probs[idx + 6] + probs[idx + 10] + probs[idx + 14]
        grammar += probs[idx] + probs[idx + 4] + probs[idx + 8] + probs[idx + 12]
        grammar += probs[idx + 19] + probs[idx + 23] + probs[idx + 27] + probs[idx + 31]
    return grammar / 2


def calc_other_bias(probs):
    """
    :param probs: list of negative log likelihoods for a corpus
    :return: gender bias in corpus
    """
    bias = 0
    for idx in range(0, len(probs), 16):
        bias -= probs[idx + 1] + probs[idx + 3] + probs[idx + 5] + probs[idx + 7]
        bias += probs[idx + 8] + probs[idx + 10] + probs[idx + 12] + probs[idx + 14]
    return bias / 4


def calc_other_grammar(probs):
    """
    :param probs: list of negative log likelihoods for a corpus
    :return: grammaticality of corpus
    """
    grammar = 0
    for idx in range(0, len(probs), 24):
        grammar -= probs[idx + 1] + probs[idx + 3] + probs[idx + 5] + probs[idx + 7]
        grammar -= probs[idx + 8] + probs[idx + 10] + probs[idx + 12] + probs[idx + 14]
        grammar += probs[idx] + probs[idx + 2] + probs[idx + 4] + probs[idx + 6]
        grammar += probs[idx + 9] + probs[idx + 11] + probs[idx + 13] + probs[idx + 15]
    return grammar / 2


def get_bias_and_grammar():
    """
    Print bias and grammaticality for spanish, french, hebrew, and italian corpora
    """
    bias = []
    grammar = []
    for lang, lang_type in [("spanish", 1), ("new_queries_old_model_french", 1),
                          ("new_queries_old_model_hebrew", 0), ("new_queries_old_model_italian", 1)]:
        prob_file_o = "../results/" + lang + "_original-initial.outlogliks"
        prob_file_s = "../results/" + lang + "_swap-initial.outlogliks"
        prob_file_d = "../results/" + lang + "_debias-initial.outlogliks"
        probs_o = get_probs(prob_file_o)
        probs_s = get_probs(prob_file_s)
        probs_d = get_probs(prob_file_d)
        if lang_type == 0:
            bias_o = calc_hebrew_bias(probs_o)
            bias_d = calc_hebrew_bias(probs_s)
            bias_s = calc_hebrew_bias(probs_d)
            grammar_o = calc_hebrew_grammar(probs_o)
            grammar_d = calc_hebrew_grammar(probs_s)
            grammar_s = calc_hebrew_grammar(probs_d)
        elif lang_type == 1:
            bias_o = calc_romance_bias(probs_o)
            bias_d = calc_romance_bias(probs_s)
            bias_s = calc_romance_bias(probs_d)
            grammar_o = calc_romance_grammar(probs_o)
            grammar_d = calc_romance_grammar(probs_s)
            grammar_s = calc_romance_grammar(probs_d)
        elif lang_type == 2:
            bias_o = calc_russian_bias(probs_o)
            bias_d = calc_russian_bias(probs_s)
            bias_s = calc_russian_bias(probs_d)
            grammar_o = calc_russian_grammar(probs_o)
            grammar_d = calc_russian_grammar(probs_s)
            grammar_s = calc_russian_grammar(probs_d)
        else:
            bias_o = calc_other_bias(probs_o)
            bias_d = calc_other_bias(probs_s)
            bias_s = calc_other_bias(probs_d)
            grammar_o = calc_other_grammar(probs_o)
            grammar_d = calc_other_bias(probs_s)
            grammar_s = calc_other_grammar(probs_d)
        bias.append([bias_o, bias_s, bias_d])
        grammar.append([grammar_o, grammar_s, grammar_d])

    print("Bias")
    for i in range(3):
        print("\\addplot coordinates {(Esp,", bias[0][i],
              ") (Fra,", bias[1][i], ") (Heb,", bias[2][i], ") (Ita,", bias[3][i], ")};")
    x = 0
    for i in range(4):
        x += bias[i][0] / bias[i][2]
        print(bias[i][0] / bias[i][2])
    print(x/4)
    print("Grammar")
    for i in range(3):
        print("\\addplot coordinates {(Esp,", grammar[0][i],
              ") (Fra,", grammar[1][i], ") (Heb,", grammar[2][i], ") (Ita,", grammar[3][i], ")};")
    x = 0
    for i in range(4):
        x += grammar[i][1] / grammar[i][2]
        print(grammar[i][1] / grammar[i][2])
    print(x/4)
