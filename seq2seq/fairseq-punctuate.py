"""Punctuate text using a pre-trained model without having to
apply BPE encoding and binarize the text first"""

# pip install subword-nmt
import sys
import os
from fairseq.models.transformer import TransformerModel

if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_path = os.path.abspath(sys.argv[1])
    else:
        sys.exit("'Model path' argument missing! Should be a directory containing checkpoint_best.pt")

    if len(sys.argv) > 2:
        data_path = os.path.abspath(sys.argv[2])
    else:
        sys.exit("'Data path' argument missing! Should be the binary data dir")

    if len(sys.argv) > 3:
        bpe_codes = os.path.abspath(sys.argv[3])
    else:
        sys.exit("'BPE codes' argument missing! Should be subword-nmt created with learn_bpe.py")

    if len(sys.argv) > 4:
        input_file = os.path.abspath(sys.argv[4])
    else:
        sys.exit("'Input text' argument missing!")

    if len(sys.argv) > 5:
        output_file = os.path.abspath(sys.argv[5])
    else:
        sys.exit("'Output text' argument missing!")

    with open(input_file, 'r') as f:
        text = f.read().strip().splitlines()

    fout = open(output_file, 'x')

    nopuncts2puncts = TransformerModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=data_path,
        bpe='subword_nmt',
        bpe_codes=bpe_codes
    )

    # Punctuate
    textout = nopuncts2puncts.translate(text)

    fout.write('\n'.join(textout))
    fout.close()
