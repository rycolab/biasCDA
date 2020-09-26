from utils.data import sample_from_sentence, get_sentence_text
from utils.reinflection import get_feats
from sigmorphon_reinflection.decode import decode_word


class SentenceConversion:
    """
    Class to convert he gender of a word in a sentence
    """
    def __init__(self, sentence, changes, use_v1, hack_v2):
        self.sentence = sentence
        self.changes = changes
        self.use_v1 = use_v1
        self.hack_v2 = hack_v2

    def _tag_value(self, is_masc):
        return 2 if is_masc else 1

    def _change_line(self, token, reinflection_model=None, device=None, decode_fn=None, decode_trg=None):
        is_masc = token.feats['Gender'].pop() == 'Masc'
        token.feats['Gender'].add('Fem' if is_masc else 'Masc')
        line = token.conll()
        parts = line.split("\t")
        for change in self.changes:
            if int(token.id) == change[0]:
                parts[2] = change[2]
        tags = get_feats(token)
        new_form = decode_word(token.lemma, tags, reinflection_model, device, decode_fn, decode_trg)
        parts[1] = new_form
        return "\t".join(parts)

    def change_forms(self, form_idxs, reinflection_model, device, decode_fn, decode_trg):
        """
        Change the forms of a selection of words in the sentence using a reinflection model

        :param form_idxs: word indices to change
        :param reinflection_model: reinflection model
        :param device: device related to reinflection model
        :param decode_fn: Decoding function
        :param decode_trg: Decoding target
        :return: UD style string of new sentence
        """
        lines = []
        if self.sentence.id:
            change_id = ""
            for change in self.changes:
                change_id += "-" + str(change[0]) + "-" + ("M" if change[-1] else "F")
            lines.append("# sent_id = " + self.sentence.id + change_id)
        change_ids = [change[0] - 1 for change in self.changes]
        for i in range(len(self.sentence)):
            token = self.sentence[i]
            line = token.conll()
            if not token.is_multiword() and int(token.id) - 1 in form_idxs + change_ids:
                if token.lemma and len(token.feats["Gender"]) == 1:
                    line = self._change_line(token, reinflection_model, device, decode_fn, decode_trg)
            lines.append(line)
        return "\n".join(lines)

    def apply(self, model, psi, reinflection_model, device, decode_fn, decode_trg):
        """
        Apply the necessary transformation to the sentence

        :param model: model to predict which words must change
        :param psi: psi parameter for model
        :param reinflection_model:
        :param device: device related to reinflection model
        :param decode_fn: Decoding function
        :param decode_trg: Decoding target
        :return: UD style string of new sentence
        """
        sample = sample_from_sentence(self.sentence, self.use_v1, self.hack_v2)
        phi = model.create_phi(sample.T, sample.pos, sample.m)

        fixes = []
        for change in self.changes:
            fixes.append((change[0], self._tag_value(change[-1])))

        best_tags = model.best_sequence(sample.T, sample.pos, psi, phi, fixes)
        original_tags = sample.m
        tags_to_change = []
        change_ids = [change[0] - 1 for change in self.changes]
        for i in range(len(original_tags)):
            if original_tags[i] != 0 and original_tags[i] != best_tags[i] and i not in change_ids:
                tags_to_change.append(i)
        sentence = self.change_forms(tags_to_change, reinflection_model, device, decode_fn, decode_trg)
        return sentence

    def apply_swap(self, reinflection_model, device, decode_fn, decode_trg):
        """
        Apply a naive swap of only the words that should be changed
        :param reinflection_model:
        :param device: device related to reinflection model
        :param decode_fn: Decoding function
        :param decode_trg: Decoding target
        :return: UD style string of new sentence
        """
        lines = []
        if self.sentence.id:
            change_id = ""
            for change in self.changes:
                change_id += "-" + str(change[0]) + "-" + ("M" if change[-1] else "F")
            lines.append("# sent_id = " + self.sentence.id + change_id)
        change_ids = [change[0] - 1 for change in self.changes]
        for i in range(len(self.sentence)):
            token = self.sentence[i]
            line = token.conll()
            if not token.is_multiword() and int(token.id) - 1 in change_ids:
                is_masc = token.feats['Gender'].pop() == 'Masc'
                token.feats['Gender'].add('Fem' if is_masc else 'Masc')
                line = self._change_line(token, reinflection_model, device, decode_fn, decode_trg)
            lines.append(line)
        return "\n".join(lines)
