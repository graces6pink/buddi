"""Step 6: verify the preprocessing chain before spending GPU hours on fitting.

The failure modes in this pipeline are almost all silent. In particular BEV is the
only input with an os.path.exists guard (process_chi3d.py:225-230), so if the npz
filenames are wrong -- e.g. BEV was run per-image, giving {name}_0.08.npz instead
of the {name}__2_0.08.npz that video mode produces -- nothing raises, every
bev_human_idx just ends up -1, and chi3d_items_for_eval quietly returns an empty
evaluation set. chi3d_eval.py then dies with a bare NameError on `thres`
(chi3d_eval.py:277) rather than saying what went wrong.

This reports the item count that Table 3 will actually be computed over.
"""

import argparse
import json
import os
import os.path as osp
import pickle
from collections import Counter

DETECTOR_KEYS = ['openpose_human_idx', 'bev_human_idx',
                 'vitpose_human_idx', 'vitposeplus_human_idx']


def check_files(proc_root, split, subject):
    subj = osp.join(proc_root, split, subject)
    expect = {
        'images_contact': ('.jpg', ''),
        'bev': ('__2_0.08.npz', ''),
        'images_contact_vitpose': ('_keypoints.json', ''),
        'images_contact_vitposeplus': ('.pkl', ''),
        'images_contact_openpose/keypoints': ('.json', ''),
    }
    print(f'--- {subject}: file counts ---')
    n_img = None
    ok = True
    for folder, (suffix, _) in expect.items():
        d = osp.join(subj, folder)
        if not osp.isdir(d):
            print(f'  {folder:36s} MISSING DIRECTORY')
            ok = False
            continue
        n = sum(1 for f in os.listdir(d) if f.endswith(suffix))
        if folder == 'images_contact':
            n_img = n
        flag = ''
        if n_img is not None and n != n_img:
            flag = f'  <-- expected {n_img}'
            ok = False
        print(f'  {folder:36s} {n:5d} files matching *{suffix}{flag}')

    # the single most likely mistake: BEV run per-image instead of video mode
    bev_dir = osp.join(subj, 'bev')
    if osp.isdir(bev_dir):
        single = sum(1 for f in os.listdir(bev_dir)
                     if f.endswith('_0.08.npz') and not f.endswith('__2_0.08.npz'))
        if single:
            print(f'  WARNING {single} files named *_0.08.npz (single underscore). '
                  'process_chi3d.py expects *__2_0.08.npz -- rerun BEV with -m video.')
            ok = False
    return ok


def check_processed(proc_root, orig_root, split, subjects):
    pkl = osp.join(proc_root, split, 'images_contact_processed.pkl')
    if not osp.exists(pkl):
        print(f'MISSING {pkl} -- run s04_process_chi3d.py')
        return False
    processed = pickle.load(open(pkl, 'rb'))
    print(f'\n--- images_contact_processed.pkl (subjects: {sorted(processed)}) ---')

    all_ok = True
    for subject in subjects:
        if subject not in processed:
            print(f'  {subject}: NOT IN PKL')
            all_ok = False
            continue

        entries = processed[subject]
        missing = Counter()
        n_complete = 0
        for key, item in entries.items():
            bad = [d for cidx in (0, 1) for d in DETECTOR_KEYS
                   if item[cidx][d] == -1]
            if bad:
                missing.update(set(bad))
            else:
                n_complete += 1

        print(f'  {subject}: {len(entries)} entries, {n_complete} with all four '
              f'detectors matched for both people')
        for d, c in missing.most_common():
            print(f'      {d:24s} == -1 in {c} entries')
        if n_complete == 0:
            all_ok = False

    # replicate the exact gate chi3d_eval.py applies
    from llib.methods.hhcs_optimization.evaluation.utils import chi3d_items_for_eval
    items = chi3d_items_for_eval(subjects, split, orig_root, processed)
    total = sum(len(cams) for s in items.values() for cams in s.values())
    print(f'\nchi3d_items_for_eval -> {total} (subject, action, camera) items '
          f'across {sum(len(v) for v in items.values())} actions')
    if total == 0:
        print('  EVALUATION SET IS EMPTY -- chi3d_eval.py will crash with NameError.')
        all_ok = False
    return all_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder', default='datasets/original/CHI3D')
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--split', default='train')
    ap.add_argument('--subjects', nargs='+', default=['s03'])
    args = ap.parse_args()

    ok = all(check_files(args.processed_data_folder, args.split, s) for s in args.subjects)
    ok &= check_processed(args.processed_data_folder, args.original_data_folder,
                          args.split, args.subjects)

    split_fn = osp.join(args.processed_data_folder, args.split, 'train_val_split.npz')
    print(f'\ntrain_val_split.npz: {"present" if osp.exists(split_fn) else "MISSING"}')
    ok &= osp.exists(split_fn)

    for p in ['essentials/body_model_utils/joint_regressors/SMPLX_to_J14.pkl',
              'essentials/body_model_utils/joint_regressors/J_regressor_h36m.npy']:
        print(f'{p}: {"present" if osp.exists(p) else "MISSING"}')
        ok &= osp.exists(p)

    print('\n' + ('ALL CHECKS PASSED' if ok else 'CHECKS FAILED'))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
