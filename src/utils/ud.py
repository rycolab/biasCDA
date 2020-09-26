
"""
Universal Dependency enumeration dictionaries
"""
# Dependency relations

__deprelv1_dict = {
    'root': 0,
    'acl': 1,
    'advcl': 2,
    'advmod': 3,
    'amod': 4,
    'appos': 5,
    'aux': 6,
    'auxpass': 7,
    'case': 8,
    'cc': 9,
    'ccomp': 10,
    'compound': 11,
    'conj': 12,
    'cop': 13,
    'csubj': 14,
    'csubjpass': 15,
    'dep': 16,
    'det': 17,
    'discourse': 18,
    'dislocated': 19,
    'dobj': 20,
    'expl': 21,
    'foreign': 22,
    'goeswith': 23,
    'iobj': 24,
    'list': 25,
    'mark': 26,
    'mwe': 27,
    'name': 28,
    'neg': 29,
    'nmod': 30,
    'nsubj': 31,
    'nsubjpass': 32,
    'nummod': 33,
    'parataxis': 34,
    'punct': 35,
    'remnant': 36,
    'reparandum': 37,
    'vocative': 38,
    'xcomp': 39
    }


__deprel_dict = {'root': 0,
               'acl': 1,
               'advcl': 2,
               'advmod': 3,
               'amod': 4,
               'appos': 5,
               'aux': 6,
               'case': 7,
               'cc': 8,
               'ccomp': 9,
               'clf': 10,
               'compound': 11,
               'conj': 12,
               'cop': 13,
               'csubj': 14,
               'dep': 15,
               'det': 16,
               'discourse': 17,
               'dislocated': 18,
               'expl': 19,
               'fixed': 20,
               'flat': 21,
               'goeswith': 22,
               'iobj': 23,
               'list': 24,
               'mark': 25,
               'nmod': 26,
               'nsubj': 27,
               'nummod': 28,
               'obj': 29,
               'obl': 30,
               'orphan': 31,
               'parataxis': 32,
               'punct': 33,
               'reparandum': 34,
               'vocative': 35,
               'xcomp': 36,
               }

# Universal Part of Speech Tags
__uposv1_dict = {'ADJ': 0,
               'ADP': 1,
               'ADV': 2,
               'AUX': 3,
               'CONJ': 4,
               'DET': 5,
               'INTJ': 6,
               'NOUN': 7,
               'NUM': 8,
               'PART': 9,
               'PRON': 10,
               'PROPN': 11,
               'PUNCT': 12,
               'SCONJ': 13,
               'SYM': 14,
               'VERB': 15,
               'X': 16
               }

__upos_dict = {'ADJ': 0,
               'ADP': 1,
               'ADV': 2,
               'AUX': 3,
               'CCONJ': 4,
               'DET': 5,
               'INTJ': 6,
               'NOUN': 7,
               'NUM': 8,
               'PART': 9,
               'PRON': 10,
               'PROPN': 11,
               'PUNCT': 12,
               'SCONJ': 13,
               'SYM': 14,
               'VERB': 15,
               'X': 16
               }

# Features
__gender_dict = {'Com': 0, 'Fem': 1, 'Masc': 2, 'Neut': 0}


def get_rel_id(rel, use_v1):
    """
    :param rel: dependency relation name
    :return: dependency relation id
    """
    dep_dict = __deprelv1_dict if use_v1 else __deprel_dict
    return dep_dict[rel.lower().split(':')[0]]


def get_rel(id, use_v1):
    """
    :param id: dependency relation id
    :return: dependency relation name
    """
    dep_dict = __deprelv1_dict if use_v1 else __deprel_dict
    for rel in dep_dict:
        if dep_dict[rel] == id:
            return rel
    raise ValueError("Id does not correspond to a Universal Dependency relation")


def get_num_rel(use_v1):
    return len(__deprelv1_dict) if use_v1 else len(__deprel_dict)


def get_upos_id(upos, use_v1):
    """
    :param upos: universal part of speech tag name
    :return: universal part of speech tag id
    """
    upos_dict = __uposv1_dict if use_v1 else __upos_dict
    if use_v1 and upos == "CCONJ":
        upos = "CONJ"
    return upos_dict[upos]


def get_upos(id, use_v1):
    """
        :param id: universal part of speech tag id
        :return: universal part of speech tag name
        """
    upos_dict = __uposv1_dict if use_v1 else __upos_dict
    for upos in upos_dict:
        if upos_dict[upos] == id:
            return upos
    raise ValueError("Id does not correspond to a Universal Dependency POS tag")


def get_num_upos(use_v1):
    return len(__uposv1_dict) if use_v1 else len(__upos_dict)


def get_gender_id(gender):
    """
    :param feat: feature name
    :param ans: feature answer
    :return: feature id, answer id
    """
    return __gender_dict[gender]


def get_gender(id):
    """
    :param id: feature id
    :return: feature dictionary
    """
    for gender in __gender_dict:
        if __gender_dict[gender] == id:
            return gender
    raise ValueError("Id does not correspond to a Universal Dependency gender")
