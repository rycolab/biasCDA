'''
Decode model
'''
import argparse
from functools import partial

import torch

from sigmorphon_reinflection.dataloader import BOS, EOS, UNK_IDX
from sigmorphon_reinflection.reinflection_model import decode_beam_search, decode_greedy
from sigmorphon_reinflection.util import maybe_mkdir
import sigmorphon_reinflection.reinflection_model as reinflection_model


def get_args():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', required=True, help='Dev/Test file')
    parser.add_argument('--out_file', required=True, help='Output file')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--max_len', default=100, type=int)
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--nonorm', default=False, action='store_true')
    return parser.parse_args()
    # yapf: enable


def setup_inference(opt):
    decode_fn = None
    if opt.decode == 'greedy':
        decode_fn = partial(decode_greedy, max_len=opt.max_len)
    elif opt.decode == 'beam':
        decode_fn = partial(
            decode_beam_search,
            max_len=opt.max_len,
            nb_beam=opt.beam_size,
            norm=not opt.nonorm)
    return decode_fn


def setup_inference_explicit(use_greedy=True, max_len=100, beam_size=5, nonorm=False):
    if use_greedy:
        decode_fn = partial(decode_greedy, max_len=max_len)
    else:
        decode_fn = partial(
            decode_beam_search,
            max_len=max_len,
            nb_beam=beam_size,
            norm=not nonorm)
    return decode_fn


def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            lemma, _, tags = line.strip().split('\t')
            yield list(lemma), tags.split(';')


def encode(model, lemma, tags, device):
    tag_shift = model.src_vocab_size - len(model.attr_c2i)

    src = []
    src.append(model.src_c2i[BOS])
    for char in lemma:
        src.append(model.src_c2i.get(char, UNK_IDX))
    src.append(model.src_c2i[EOS])

    attr = [0] * (len(model.attr_c2i) + 1)
    for tag in tags:
        if tag in model.attr_c2i:
            attr_idx = model.attr_c2i[tag] - tag_shift
        else:
            attr_idx = -1
        if attr[attr_idx] == 0:
            attr[attr_idx] = model.attr_c2i.get(tag, 0)

    return (torch.tensor(src, device=device).view(len(src), 1),
            torch.tensor(attr, device=device).view(1, len(attr)))


def get_decoding_model(model_file, use_greedy=True, max_len=100, beam_size=5, nonorm=False):
    with torch.no_grad():
        decode_fn = setup_inference_explicit(use_greedy, max_len, beam_size, nonorm)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(open(model_file, mode='rb'), map_location=device)
        model = model.to(device)

        trg_i2c = {i: c for c, i in model.trg_c2i.items()}
        decode_trg = lambda seq: [trg_i2c[i] for i in seq]

        return model, device, decode_fn, decode_trg


def decode_word(lemma, tags, model, device, decode_fn, decode_trg):
    src = encode(model, lemma, tags, device)
    pred, _ = decode_fn(model, src)
    return ''.join(decode_trg(pred))


def main():
    opt = get_args()

    decode_fn = setup_inference(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(open(opt.model, mode='rb'), map_location=device)
    model = model.to(device)

    trg_i2c = {i: c for c, i in model.trg_c2i.items()}
    decode_trg = lambda seq: [trg_i2c[i] for i in seq]

    maybe_mkdir(opt.out_file)
    with open(opt.out_file, 'w', encoding='utf-8') as fp:
        for lemma, tags in read_file(opt.in_file):
            src = encode(model, lemma, tags, device)
            pred, _ = decode_fn(model, src)
            pred_out = ''.join(decode_trg(pred))
            fp.write("".join(lemma) + '\t' + pred_out + '\t' + ";".join(tags[1:]) + '\n')


if __name__ == '__main__':
    with torch.no_grad():
        main()
