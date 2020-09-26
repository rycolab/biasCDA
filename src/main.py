import argparse
from animacy import get_animate_samples
from model import Model
from sigmorphon_reinflection.decode import get_decoding_model
from sigmorphon_reinflection.reinflection_model import *
from tqdm import tqdm
from utils.conll import load_sentences

def get_args():
    """
    :return: command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_files', required=True, nargs='+', help='Input conllu file')
    parser.add_argument('--out_file', required=True, help='Output conllu file')
    parser.add_argument('--psi', required=True, help='Path to psi parameters')
    parser.add_argument('--reinflect', required=True, help='Path to reinflection model')
    parser.add_argument('--animate_list', required=True, help='Path to animate noun list')
    parser.add_argument('--inc_input', default=False, action='store_true', help='True if input should be copied into output file')
    parser.add_argument('--use_v1', default=False, action='store_true')
    parser.add_argument('--hack_v2', default=False, action='store_true')
    parser.add_argument('--get_ids', default=False, action='store_true')
    parser.add_argument('--part', type=int)
    return parser.parse_args()


def main():
    """
    Program to convert all animate nouns in a UD annotated corpus
    """
    # Get command line arguments
    opt = get_args()
    # Load models
    model = Model([0, 1, 2])
    psi = torch.load(opt.psi)
    print(psi.shape)
    with torch.no_grad():
        reinflection_model, device, decode_fn, decode_trg = get_decoding_model(opt.reinflect)
    print("Models loaded")

    out = open(opt.out_file, "w")
    if not isinstance(opt.in_files, list):
        opt.in_files = [opt.in_files]
    count = 0
    # Find and convert sentences with animate nouns for each file
    for i in range(len(opt.in_files)):
        file = opt.in_files[i]
        print("Processing file " + str(i + 1) + " out of " + str(len(opt.in_files)) + " files")
        part = 1
        with open(file, "r") as f:
            # line = f.readline()
            # Work in batches of 100,000 sentences to avoid memory issues
            not_empty = True
            while not_empty:
                print("  Partition", part)
                # Load sentences
                print("    Loading partition...")
                conll, not_empty = load_sentences(10000, f)
                # Extract sentences with animate nouns
                print("    Finding animate nouns...")
                samples = get_animate_samples(conll, opt.animate_list, opt.use_v1, opt.hack_v2)
                if opt.inc_input:
                    out.write(conll.conll())
                del conll
                print("     ", str(len(samples)), "animate nouns found")

                # Convert gender of sentences
                converted_sentences = []
                print("    Converting sentences...")
                for sc in tqdm(samples, total=len(samples)):
                    try:
                        converted = sc.apply(model, psi, reinflection_model, device, decode_fn, decode_trg)
                        converted_sentences.append(converted)
                    except ValueError:
                        continue
                    except IndexError:
                        continue
                del samples
                out.write("\n\n".join(converted_sentences) + "\n\n")
                del converted_sentences
                part += 1
                if opt.part and opt.part < part:
                    break
    out.close()
    print("Done")


if __name__ == '__main__':
    main()
