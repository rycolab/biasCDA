from utils.ud import get_rel_id, get_upos_id, get_gender_id


class Sentence:
    """
    Object o maintain information about a sentence
    """
    def __init__(self, T, pos, m_o, m_m=None, i=None, from_masc=None, sent_id=None):
        self.T = T
        self.pos = pos
        self.m = m_o
        self.m_eval = m_m
        self.i = i
        self.from_masc = from_masc
        self.eval_sent = True if i else False
        self.sent_id = sent_id


def sample_from_sentence(sent, use_v1, hack_v2):
    """
    Get dependency tree and tag vector from a sentence
    :param sent: UD parsed sentence
    :param use_v1: True if sentence is annotated using UD V1.2
    :return: dependency tree, tag vector
    """
    T = []
    pos = []
    m = []
    for tok in sent:
        try:
            idx = int(tok.id)
        except ValueError:
            continue
        deprel = tok.deprel
        upos = tok.upos
        if not use_v1 and hack_v2:
            if upos == "CONJ":
                upos = "CCONJ"
            if deprel == "cop":
                upos = "AUX"
            elif deprel == "neg":
                deprel = "advmod"
            elif deprel == "name" or deprel == "foreign":
                deprel = "flat"
            elif deprel == "mwe":
                deprel = "fixed"
            elif deprel == "dobj":
                deprel = "obj"
            if "pass" in deprel:
                deprel = deprel.replace("pass", ":pass")
        T.append((idx, int(tok.head), get_rel_id(deprel, use_v1)))
        pos.append(get_upos_id(upos, use_v1))
        tag = 0 if ('Gender' not in tok.feats or (tok.feats['Gender'] != {"Masc"} and tok.feats['Gender'] != {"Fem"}))\
            else get_gender_id(next(iter(tok.feats['Gender'])))
        m.append(tag)
    return Sentence(T, pos, m)


def samples_from_conll(conll, use_v1, hack_v2):
    """
    Get dependency trees and tag vectors from a collection of sentences
    :param conll: collection of UD parsed sentences
    :param use_v1: True if sentence is annotated using UD V1.2
    :return: samples
    """
    samples = []
    for sent in conll:
        samples.append(sample_from_sentence(sent, use_v1, hack_v2))
    return samples


def get_tags(samples):
    """
    :param samples: samples used for training
    :return: all the tags that occur in samples
    """
    tags = []
    for sentence in samples:
        for t in sentence.m:
            if t not in tags:
                tags.append(t)
    tags.sort()
    return tags


def eval_samples_from_conll(conll, use_v1, hack_v2):
    """
    :param conll: PyConll object
    :param use_v1: True if sentence is annotated using UD V1.2
    :return: list of evaluation samples
    """
    eval_dict = dict()
    for sent in conll:
        sent_id, idx, gender, orig = sent.id.split('-')
        sample = sample_from_sentence(sent, use_v1, hack_v2)
        if sent_id not in eval_dict:
            from_masc = (orig == 'o' and gender == 'M') or (orig == 'm' and gender == 'F')
            eval_dict[sent_id] = {'T': sample.T, 'pos': sample.pos, 'from_masc': from_masc, 'idx': int(idx)}
        if orig == 'o':
            eval_dict[sent_id]['m_o'] = sample.m
        else:
            eval_dict[sent_id]['m_m'] = sample.m
    eval_samples = []
    for sent_id in eval_dict:
        sample = eval_dict[sent_id]
        eval_samples.append(Sentence(sample['T'], sample['pos'], sample['m_o'],
                                     sample['m_m'], sample['idx'], from_masc, sent_id))
    return eval_samples


def get_sentence_text(sentence):
    """
    :param sentence: UD sentence
    :return: text version of the UD sentence
    """
    text = ""
    for token in sentence:
        if not token.is_multiword() and token.form:
            text += token.form + " "
    return text[:-1]
