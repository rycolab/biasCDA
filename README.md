# Mitigating Gender Bias by Counterfactual Data Augmentation

## Citation
This code is for the paper
_Counterfactual data augmentation for mitigating gender stereotypes in languages with rich morphology_
featured in ACL 2019. Please cite as:
```bibtex
@inproceedings{zmigrod-etal-2019-counterfactual,
    title = "Counterfactual Data Augmentation for Mitigating Gender Stereotypes in Languages with Rich Morphology",
    author = "Zmigrod, Ran  and
      Mielke, Sabrina J.  and
      Wallach, Hanna  and
      Cotterell, Ryan",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1161",
    doi = "10.18653/v1/P19-1161",
    pages = "1651--1661"
}
```
## Requirements
* Python version >= 3.6
* Pytorch

## Running and Training
We provide pre-trained models for French and Spanish in the `models` folder.
All input files should be in conllu format.

In order to run a pretrained model, use the command.
```bash
python src/main.py --in_files [input conllu files] --psi [path to psi .pt file] --reinflect [path to reinflectino model] --animate_list [path to animacy list] --inc_input --get_ids  --out_file [path to output_file] --part 100
```
In order to train the model, use the following command
```bash
python src/neural-mrf.py --data [path to training data] --out_dir [path to output directory]--log_alpha 1 --lr 0.005 --wd 0.0001
```
You can train the reinflection using `reinflection_train.py`.
This has been lightly modified by the [Sigmorphon cross-lingual-baseline](https://github.com/sigmorphon/crosslingual-inflection-baseline).
If you use this code please cite the shared task appropriately.