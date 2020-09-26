from utils.data import get_sentence_text
from SentenceConversion import SentenceConversion
from copy import deepcopy
from tqdm import tqdm
from itertools import combinations
import pickle


def get_fem_word(masc_word, words):
    """
    :param masc_word: masculine word to convert
    :param words: list of English-Femining-Masculine word triples
    :return: feminine inflection of masculine word
    """
    for _, fem_word, word in words:
        if masc_word == word:
            return fem_word
    raise ValueError(masc_word)


def get_masc_word(fem_word, words):
    """
    :param fem_word: feminine word to convert
    :param words: list of English-Femining-Masculine word triples
    :return: masculine inflection of feminine word
    """
    for _, word, masc_word in words:
        if fem_word == word:
            return masc_word
    raise ValueError(fem_word)


def get_animate_samples(conll, animate_file, use_v1, hack_v2):
    """
    :param conll: conll object
    :param animate_file: file containing animate noun pairs
    :param use_v1: True if sentence is annotated using UD V1.2
    :param hack_v2: True if sentence should be made into UD V2 from V1.2
    :return: list of SentenceConversion objects
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    words = [line[1] for line in lines] + [line[2] for line in lines]

    samples = []
    for sent in tqdm(conll, total=len(conll)):
        changes = []
        for tok in sent:
            if tok.upos != "NOUN" or 'Gender' not in tok.feats or len(tok.feats['Gender']) != 1:
                continue
            is_masc = next(iter(tok.feats['Gender'])) == 'Masc'
            if tok.lemma in words:
                try:
                    convert = get_fem_word(tok.lemma, lines) if is_masc else get_masc_word(tok.lemma, lines)
                except ValueError:
                    continue
                changes.append((int(tok.id), convert, convert, not is_masc))
        changes_list = []
        for r in range(1, len(changes) + 1):
            changes_list.extend(combinations(changes, r))
        for change in changes_list:
            samples.append(SentenceConversion(deepcopy(sent), change, use_v1, hack_v2))
    return samples


def get_animate_sentences(conll, animate_file):
    """
    :param conll: conll object
    :param animate_file: file containing animate noun pairs
    :return: list of tab separated conversion sentence entries
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    words = [line[1] for line in lines] + [line[2] for line in lines]

    conversion = []
    for sent in conll:
        for tok in sent:
            if tok.lemma in words and tok.upos == "NOUN" and 'Gender' in tok.feats and len(tok.feats['Gender']) == 1:
                gender = next(iter(tok.feats['Gender']))
                try:
                    convert = get_fem_word(tok.lemma, lines) if gender == 'Masc' else get_masc_word(tok.lemma, lines)
                except ValueError:
                    continue
                conversion.append("\t".join([sent.id, tok.id, tok.lemma, convert, get_sentence_text(sent)]) + "\n")
    return conversion


def write_heb_queries(animate_file, out_file):
    """
    Write Hebrew query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    adj_sg = [
        ["טוב", "טובה"],
        ["רע", "רעה"],
        ["חכם", "חכמה"],
        ["יפה", "יפה"]
    ]
    adj_pl = [
        ["טובים", "טובות"],
        ["רעים", "רעות"],
        ["חכמים", "חכמות"],
        ["יפים", "יפות"]
    ]
    queries = ""
    for line in lines:
        fem, masc = line[1], line[2]
        for adj in adj_sg:
            for a in adj:
                queries += " ".join(["ה", fem, "ה", a]) + "\n"
                queries += " ".join(["ה", masc, "ה", a]) + "\n"
        if fem.endswith("ה") or fem.endswith("ת"):
            fem_pl = fem[:-1]
        elif fem.endswith("ם"):
            fem_pl = fem[:-1] + "מ"
        elif fem.endswith("ן"):
            fem_pl = fem[:-1] + "נ"
        elif fem.endswith("ף‬"):
            fem_pl = fem[:-1] + "פ‬"
        elif fem.endswith("ץ"):
            fem_pl = fem[:-1] + "צ"
        else:
            fem_pl = fem
        fem_pl += "ות"
        if masc.endswith("ה") or masc.endswith("י"):
            masc_pl = masc[:-1]
        elif masc.endswith("ם"):
            masc_pl = masc[:-1] + "מ"
        elif masc.endswith("ן"):
            masc_pl = masc[:-1] + "נ"
        elif masc.endswith("ף‬"):
            masc_pl = masc[:-1] + "פ‬"
        elif fem.endswith("ץ"):
            masc_pl = masc[:-1] + "צ"
        else:
            masc_pl = masc
        masc_pl += "ים"
        for adj in adj_pl:
            for a in adj:
                queries += " ".join(["ה", fem_pl, "ה", a]) + "\n"
                queries += " ".join(["ה",  masc_pl, "ה", a]) + "\n"
    with open(out_file, "w") as f:
        f.write(queries)


def write_spanish_queries(animate_file, out_file):
    """
    Write Spanish query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    queries = ""
    adj_sg = [
        ["bueno", "buena"],
        ["mal", "mala"],
        ["inteligente", "inteligente"],
        ["hermoso", "hermosa"]
    ]
    adj_pl = [
        ["buenos", "buenas"],
        ["malos", "malas"],
        ["inteligentes", "inteligentes"],
        ["hermosos", "hermosas"]
    ]
    det_sg, det_pl = ["El", "La"], ["Los", "Las"]
    for line in lines:
        fem, masc = line[1], line[2]
        for det in det_sg:
            for adj in adj_sg:
                for a in adj:
                    queries += " ".join([det, fem, a]) + "\n"
                    queries += " ".join([det, masc, a]) + "\n"
        fem_pl = fem + ('s' if fem.endswith(('a', 'e', 'i', 'o', 'u')) else 'es')
        masc_pl = masc + ('s' if masc.endswith(('a', 'e', 'i', 'o', 'u')) else 'es')
        for det in det_pl:
            for adj in adj_pl:
                for a in adj:
                    queries += " ".join([det, fem_pl, a]) + "\n"
                    queries += " ".join([det, masc_pl, a]) + "\n"
    with open(out_file, "w") as f:
        f.write(queries)


def write_french_queries(animate_file, out_file):
    """
    Write French query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    queries = ""
    adj_sg = [
        ["bon", "bonne"],
        ["mauvais", "mauvaise"],
        ["intelligent", "intelligente"],
        ["bel", "belle"]
    ]
    adj_pl = [
        ["bons", "bonnes"],
        ["mauvais", "mauvaises"],
        ["intelligents", "intelligentes"],
        ["beaux", "belles"]
    ]
    det_sg, det_pl = ["Le", "La"], ["Les", "Les"]
    for line in lines:
        fem, masc = line[1], line[2]
        for det in det_sg:
            for adj in adj_sg:
                for a in adj:
                    queries += " ".join([det, fem, a]) + "\n"
                    queries += " ".join([det, masc, a]) + "\n"
        if fem.endswith("s") or fem.endswith("x") or fem.endswith("z"):
            fem_pl = fem
        elif fem.endswith("eau") or fem.endswith("eu"):
            fem_pl = fem + "x"
        elif fem.endswith("al"):
            fem_pl = fem[:-1] + "ux"
        elif fem.endswith("ail"):
            fem_pl = fem[:-2] + "ux"
        else:
            fem_pl = fem + "s"
        if masc.endswith("s") or masc.endswith("x") or masc.endswith("z"):
            masc_pl = masc
        elif masc.endswith("eau") or masc.endswith("eu"):
            masc_pl = masc + "x"
        elif masc.endswith("al"):
            masc_pl = masc[:-1] + "ux"
        elif masc.endswith("ail"):
            masc_pl = masc[:-2] + "ux"
        else:
            masc_pl = masc + "s"
        for det in det_pl:
            for adj in adj_pl:
                for a in adj:
                    queries += " ".join([det, fem_pl, a]) + "\n"
                    queries += " ".join([det, masc_pl, a]) + "\n"
    with open(out_file, "w") as f:
        f.write(queries)


def write_italian_queries(animate_file, out_file):
    """
    Write Italian query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    queries = ""
    adj_sg = [
        ["buono", "buona"],
        ["cattivo", "cattiva"],
        ["intelligente", "intelligenti"],
        ["bello", "bella"]
    ]
    adj_pl = [
        ["buoni", "buone"],
        ["cattivi", "cattive"],
        ["intelligenti", "intelligenti"],
        ["belli", "belle"]
    ]
    det_sg, det_pl = ["Lo", "La"], ["Gli", "Les"]
    for line in lines:
        fem, masc = line[1], line[2]
        for det in det_sg:
            for adj in adj_sg:
                for a in adj:
                    queries += " ".join([det, fem, a]) + "\n"
                    queries += " ".join([det, masc, a]) + "\n"
        if fem.endswith("a"):
            fem_pl = fem[:-1] + "e"
        elif fem.endswith("ie"):
            fem_pl = fem
        elif fem.endswith("e") or fem.endswith("o") or fem.endswith("i"):
            fem_pl = fem[:-1] + "i"
        else:
            continue
        masc_pl = masc[:1] + "i"
        for det in det_pl:
            for adj in adj_pl:
                for a in adj:
                    queries += " ".join([det, fem_pl, a]) + "\n"
                    queries += " ".join([det, masc_pl, a]) + "\n"
    with open(out_file, "w") as f:
        f.write(queries)


def write_russian_queries(animate_file, out_file):
    """
    Write Russian query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    print(len(lines))
    queries = ""
    adj_sg = [
        ["Xороший", "Xорошая"],
        ["Плохой", "Плохая"],
        ["Yмный", "Yмная"],
        ["Красивый", "Красивая"]
    ]
    adj_pl = ["Xорошие", "Плохие", "Yмные", "Красивые"]
    bad = []
    for line in lines:
        fem, masc = line[1], line[2]
        for adj in adj_sg:
            for a in adj:
                queries += " ".join([a, fem]) + "\n"
                queries += " ".join([a, masc]) + "\n"
        if not fem.endswith(('а', 'ия', 'г', 'к', 'х', 'ж', 'ч', 'ш', 'щ', 'я', 'ь')):
            bad.append((fem, masc))
        if fem.endswith('а'):
            fem_pl = fem[:-1] + "ы"
        elif fem.endswith('ия'):
            fem_pl = fem[:-2] + "ии"
        elif fem.endswith(('г', 'к', 'х', 'ж', 'ч', 'ш', 'щ', 'я', 'ь')):
            fem_pl = fem[:-1] + "и"
        else:
            fem_pl = fem + 'ы'
        if masc.endswith('тель'):
            masc_pl = masc[:-4] + 'и'
        elif masc.endswith(('й', 'ь', 'г', 'к', 'х', 'ж', 'ч', 'ш', 'щ')):
            masc_pl = masc[:-4] + 'и'
        else:
            masc_pl = masc + 'ы'
        for a in adj_pl:
            queries += " ".join([a, fem_pl]) + "\n"
            queries += " ".join([a, masc_pl]) + "\n"
    with open("bad.pickle", "wb") as f:
        pickle.dump(bad, f)
    with open(out_file, "w") as f:
        f.write(queries)


def write_polish_queries(animate_file, out_file):
    """
    Write Polish query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    queries = ""
    adj_sg = [
        ["Dobry", "Dobra"],
        ["Zły", "Zła"],
        ["Mądry", "Mądra"],
        ["Piękny", "Piękna"]
    ]
    adj_pl = [
        ["Dobrzy", "Dobre"],
        ["Źli", "Złe"],
        ["Mądrzy", "Mądre"],
        ["Piękni", "Piękne"]
    ]
    bad = []
    for line in lines:
        fem, masc = line[1], line[2]
        if not fem.endswith(('kа', 'ga', 'ca', 'ea', 'ia', 'ja', 'la', 'ża', 'rza', 'a')):
            bad.append((fem, masc))
        if fem.endswith(('kа', 'ga')):
            fem_pl = fem[:-1] + "i"
        elif fem.endswith(('ca', 'ea', 'ia', 'ja', 'la', 'ża', 'rza')):
            fem_pl = fem[:-1] + "e"
        elif fem.endswith('a'):
            fem_pl = fem[:-1] + "y"
        else:
            fem_pl = fem[:-1] + "y"
        if masc.endswith(('k', 'g')):
            masc_pl = masc + 'i'
        elif masc.endswith(('c', 'j', 'l', 'ż', 'rz')):
            masc_pl = masc + 'e'
        elif masc.endswith(('ć', 'ń', 'ś', 'ź')):
            masc_pl = masc + 'ie'
        else:
            masc_pl = masc + 'y'
        for adj in adj_sg:
            for a in adj:
                queries += " ".join([a, fem]) + "\n"
                queries += " ".join([a, masc]) + "\n"
        for adj in adj_pl:
            for a in adj:
                queries += " ".join([a, fem_pl]) + "\n"
                queries += " ".join([a, masc_pl]) + "\n"
    with open("bad2.pickle", "wb") as f:
        pickle.dump(bad, f)
    with open(out_file, "w") as f:
        f.write(queries)


def write_german_queries(animate_file, out_file):
    """
    Write German query list from list of animate noun
    :param animate_file: file containing animate nouns
    :param out_file: output file
    """
    with open(animate_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    queries = ""
    adj_sg = ["gute", "schlechte", "schlaue", "schöne"]
    det = ["Der", "Die"]
    for line in lines:
        fem, masc = line[1], line[2]
        for d in det:
            for a in adj_sg:
                queries += " ".join([d, a, fem.capitalize()]) + "\n"
                queries += " ".join([d, a, masc.capitalize()]) + "\n"
    with open(out_file, "w") as f:
        f.write(queries)

