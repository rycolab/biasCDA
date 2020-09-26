import argparse
from utils.conll import load_sentences
from utils.data import get_sentence_text
from tqdm import tqdm
import os

"""
Program to convert a conll file to a text file
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_files', required=True, nargs='+', help='Input conllu files')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--part', type=int)
    opt = parser.parse_args()

    for i in range(len(opt.in_files)):
        file = opt.in_files[i]
        out_file = os.path.join(opt.out_dir, file.split("/")[-1][:file.rfind('.')] + "_text")
        out = open(out_file, "w")
        print("Processing file " + str(i + 1) + " out of " + str(len(opt.in_files)) + " files")
        part = 1
        with open(file, "r") as f:
            not_empty = True
            while not_empty and (not opt.part or part <= opt.part):
                print("  Partition", part)
                conll, not_empty = load_sentences(10000, f)
                out_text = ""
                for sent in tqdm(conll, total=len(conll)):
                    try:
                        out_text += get_sentence_text(sent) + "\n"
                    except TypeError:
                        for x in sent:
                            print(x.id, x.form, x.lemma, x.upos, x.head, x.deprel)
                        exit()
                del conll
                out.write(out_text)
                del out_text
                part += 1
        out.close()
    print("Done")
