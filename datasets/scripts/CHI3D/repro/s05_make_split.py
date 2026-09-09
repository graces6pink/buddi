"""Step 5: write datasets/processed/CHI3D/train/train_val_split.npz.

No script in the repo generates this file, yet chi3d.py:59-61 and
chi3d_eval.py:48 both require it. It holds two arrays of *subject folder names*
under the keys 'train' and 'val'; --eval-split selects one of them.

The paper uses "the contact frame of the sequences from two subject pairs ... and
the third pair for evaluation" and Table 3 is labelled "pair s03", so the split is
train=[s02, s04], val=[s03].

Note the file goes under train/, not the CHI3D root that DATA.md:172-175 shows --
the docs are wrong here, the code is authoritative.
"""

import argparse
import os
import os.path as osp

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--split', default='train')
    ap.add_argument('--train-subjects', nargs='+', default=['s02', 's04'])
    ap.add_argument('--val-subjects', nargs='+', default=['s03'])
    args = ap.parse_args()

    out_dir = osp.join(args.processed_data_folder, args.split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, 'train_val_split.npz')

    np.savez(out_path,
             train=np.array(args.train_subjects),
             val=np.array(args.val_subjects))
    check = np.load(out_path)
    print(f'wrote {out_path}')
    print(f"  train = {list(check['train'])}")
    print(f"  val   = {list(check['val'])}")


if __name__ == '__main__':
    main()
