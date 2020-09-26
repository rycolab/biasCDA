import re
from utils.data import get_plural_noun, get_singular_noun
from pyconll import load_from_string
from string import punctuation


def convert_to_conll(sent_conll, old, new, idx, lemma, is_masc):
    """
    :param sent_conll: conll sentence file to modify
    :param old: original sentence
    :param new: new sentence
    :param idx: id of changed word
    :param lemma: lemma of changed word
    :param is_masc: True if word to change is masculine, False otherwise
    :return: conll text for a modified conll sentence
    """
    old = old.replace(" del ", " de el ").replace(" al ", " a el ")
    new = new.replace(" del ", " de el ").replace(" al ", " a el ")
    old_words = old.split()
    new_words = new.split()
    old_words = [word.strip(punctuation) for word in old_words]
    new_words = [word.strip(punctuation) for word in new_words]
    old_words = [word for word in old_words if word]
    new_words = [word for word in new_words if word]
    assert len(old_words) == len(new_words), "\n" + str(old_words) + "\n" + str(new_words)
    word_idx = 0
    old_word = old_words[word_idx]
    new_word = new_words[word_idx]
    conll_text = ""
    if sent_conll.id:
        conll_text += "# sent_id = " + sent_conll.id + "-" + str(idx) + "-" + ("M" if is_masc else "F") + "\n"
    for tok in sent_conll:
        line = tok.conll()
        if tok.form.strip(punctuation) == old_word:
            if old_word != new_word and 'Gender' in tok.feats:
                is_masc = tok.feats['Gender'].pop() == 'Masc'
                tok.feats['Gender'].add('Fem' if is_masc else 'Masc')
                line = tok.conll()
                tok.feats['Gender'].pop()
                tok.feats['Gender'].add('Fem' if not is_masc else 'Masc')
                parts = line.split("\t")
                if int(tok.id) == idx:
                    parts[1], parts[2] = new_word, lemma
                else:
                    parts[1] = new_word
                line = "\t".join(parts)
            word_idx += 1
            if word_idx < len(old_words):
                old_word = old_words[word_idx]
                new_word = new_words[word_idx]
        conll_text += line + "\n"
    return conll_text


def convert_from_annotations(annotation_file, animate_file, conlls, out_file):
    """
    Create gold standard conll file
    :param annotation_file: file containing original and gold modified sentences
    :param animate_file: file containing animate noun list
    :param conlls: conll objects containing sentences
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        animate_lines = f.readlines()
    animate_lines = [line.strip().split("\t") for line in animate_lines]
    masc_words = [line[2] for line in animate_lines]
    fem_words = [line[1] for line in animate_lines]

    masc_words_plural = [get_plural_noun(word) for word in masc_words]
    fem_words_plural = [get_plural_noun(word) for word in fem_words]
    with open(annotation_file, "r") as f:
        lines = [line.strip().split("\t") for line in f.readlines()[1:]]
    out_lines = ""

    conll_i = 0
    conll = conlls[conll_i]

    i = 0
    sent = conll[i]
    for j in range(len(lines)):
        line = lines[j]
        if len(line) > 1:
            sent_id = line[0]
            while sent.id != sent_id and i < len(conll) - 1:
                i += 1
                sent = conll[i]
            if sent.id == sent_id:
                form = line[2]
                if form in masc_words_plural:
                    is_masc = True
                    lemma = get_singular_noun(form, animate_lines, is_masc)
                elif form in fem_words_plural:
                    is_masc = False
                    lemma = get_singular_noun(form, animate_lines, is_masc)
                else:
                    lemma = form
                    is_masc = form in masc_words
                out_lines += convert_to_conll(sent, line[3], line[4], int(line[1]), form, lemma, is_masc)
                out_lines += "\n"
            else:
                conll_i += 1
                if conll_i < len(conlls):
                    conll = conlls[conll_i]
                    i = 0
                    j -= 1
                else:
                    break
    with open(out_file, "w") as f:
        f.write(out_lines)
