from pyconll import load_from_file


def get_feats(token):
    """
    :param token: PyConll token
    :return: list of features of token
    """
    feats = [token.upos]
    for f in token.feats:
        for fs in token.feats[f]:
            if fs == "Yes":
                feats.append(f)
            else:
                feats.append(fs)
    return feats


def get_lines(conll):
    """
    :param conll: PyConll object
    :return: list of lemma-form-feature triples
    """
    lines = []
    words = set()
    for sent in conll:
        for tok in sent:
            if not tok.form:
                continue
            form = tok.form.lower()
            if (form, tok.lemma) in words or not tok.lemma or not tok.feats or 'Gender' not in tok.feats or\
                tok.upos == "PROPN" or "_" in form:
                continue
            else:
                feats = get_feats(tok)
                lines.append(tok.lemma + "\t" + tok.form.lower() + "\t" + ";".join(feats) + "\n")
                words.add((tok.form, tok.lemma))
    return lines


def create_reinflection_file(in_file, out_file):
    """
    :param in_file: UD annotated input file name
    :param out_file: output file name
    """
    conll = load_from_file(in_file)
    lines = get_lines(conll)
    with open(out_file, "w") as f:
        f.writelines(lines)
