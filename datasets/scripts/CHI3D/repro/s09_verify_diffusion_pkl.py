"""Step 9: verify {train,val}_diffusion.pkl before anything trains on it.

Every failure this catches is silent -- nothing raises, the numbers just come
out wrong and you find out from a model that trained on the wrong data.

  contact frame count off
      The expectation is computed from the annotations by expected_contact_frames()
      below, which reproduces chi3d.py's own frame_ids arithmetic -- so it already
      excludes the five sequences that expression silently drops. Measured:
      980 for train (s02+s04) and 492 for val (s03). A short count beyond that
      means images_contact_processed.pkl is missing a subject: s04_process_chi3d.py
      merges into whatever is already there rather than overwriting, so a subject
      that failed earlier in the chain (frame extraction, ViTPose, BEV) just never
      appears. Re-read s06's output.

  a subject on the wrong side of the split
      train must be exactly {s02, s04} and val exactly {s03}. s03 is the only
      clean judge we have -- CHI3D val is trustworthy precisely because no s03
      frame is ever trained on -- and if it leaks into train the metric does not
      break, it just quietly looks better than it should. train_val_split.npz
      has no upstream generator (s05_make_split.py writes it), so it is exactly
      the kind of file that gets clobbered. Fix: rerun s05, then DELETE the
      {train,val}_diffusion.pkl caches, which do not notice the split changed.

  camera views collapsed to one
      all four ids should appear at roughly a quarter each. If only 58860488 is
      present, load_single_camera or load_unit_glob_and_transl was set True
      (chi3d.py:456-457 filters on either), and the conditional model just lost
      three quarters of its viewpoint diversity.

  BEV filter attrition
      chi3d.py:465-469 unconditionally drops samples whose bev_human_idx is -1.
      Measured: 34 of 492 on val and 69 of 980 on train, both ~7%. Much worse means the
      BEV run did not produce what process_chi3d.py expects -- most likely the
      npz filenames, which must have the double underscore that `bev -m video`
      produces ({name}__2_0.08.npz, process_chi3d.py:204). Wrong names fail
      silently into all-zero placeholders.

Exit code is non-zero if any check fails, so this can gate the next stage.
"""

import argparse
import json
import os.path as osp
import pickle
import sys
from collections import Counter

import numpy as np

EXPECTED_SUBJECTS = {'train': {'s02', 's04'}, 'val': {'s03'}}
NUM_CAMERAS = 4
MAX_BEV_ATTRITION = 0.15   # s03 measured 6.9%; well above that means a broken BEV run


def expected_contact_frames(original_data_folder, subjects):
    """How many contact frames chi3d.py will actually emit, per subject.

    Not simply len(annotations) * 4. chi3d.py:327-330 builds the frame list as

        list(arange(fr_id, start_fr, -5)[::-1]) + list(arange(fr_id, end_fr-1, 5)[1:])

    and only marks is_contact_frame where frame_id == fr_id. When
    start_fr >= fr_id the first arange is empty and the second one's [1:] drops
    fr_id, so the annotated contact frame never enters the list and the whole
    sequence contributes nothing. The len(frame_ids)==0 fallback does not fire,
    because the second part is non-empty. Five sequences across s02/s03/s04 hit
    this.

    Reproducing the same arithmetic here means this check reports the real
    expectation instead of accusing the merge of losing a subject, and it keeps
    working if that upstream expression is ever fixed.
    """
    total, dropped = 0, []
    for subject in sorted(subjects):
        fn = osp.join(original_data_folder, 'train', subject,
                      'interaction_contact_signature.json')
        for action, anno in json.load(open(fn)).items():
            fr, start, end = anno['fr_id'], anno['start_fr'], anno['end_fr']
            frame_ids = (list(np.arange(fr, start, -5)[::-1])
                         + list(np.arange(fr, end - 1, 5)[1:]))
            if len(frame_ids) == 0:
                frame_ids = [fr]
            if fr in frame_ids:
                total += NUM_CAMERAS
            else:
                dropped.append(f'{subject}/{action} (fr_id {fr}, start_fr {start})')
    return total, dropped


def check_split(processed_data_folder, split, original_data_folder):
    fn = osp.join(processed_data_folder, f'{split}_diffusion.pkl')
    if not osp.exists(fn):
        print(f'[FAIL] {fn} does not exist -- run s08_make_diffusion_pkl.py')
        return False

    data = pickle.load(open(fn, 'rb'))
    contact = [x for x in data if x['is_contact_frame']]
    kept = [x for x in contact if not np.any(x['bev_human_idx'] == -1)]

    # .../CHI3D/<split_folder>/<subject>/<image_folder>/<file>
    subjects = Counter(x['imgpath'].split(osp.sep)[-3] for x in kept)
    cameras = Counter(x['imgname'].rsplit('_', 1)[1].rsplit('.', 1)[0] for x in kept)

    expected_subjects = EXPECTED_SUBJECTS[split]
    expected_contact, dropped_seqs = expected_contact_frames(
        original_data_folder, expected_subjects)
    attrition = 1 - len(kept) / len(contact) if contact else 1.0

    print(f'\n=== {split}_diffusion.pkl ===')
    print(f'  raw items          {len(data)}')
    print(f'  contact frames     {len(contact)}  (expected {expected_contact})')
    if dropped_seqs:
        print(f'  note: {len(dropped_seqs)} sequence(s) yield no contact frame because of the '
              f'chi3d.py:327-330 frame_ids expression, so they are excluded from the '
              f'expectation above:')
        for d in dropped_seqs:
            print(f'         {d}')
    print(f'  after BEV filter   {len(kept)}  (dropped {len(contact) - len(kept)}, '
          f'{attrition * 100:.1f}%)')
    print(f'  subjects           {dict(subjects)}  (expected {sorted(expected_subjects)})')
    print(f'  cameras            {dict(cameras)}')

    ok = True
    if len(contact) != expected_contact:
        print(f'  [FAIL] contact frame count is {len(contact)}, expected {expected_contact} '
              f'-> images_contact_processed.pkl is missing a subject; check s06 output')
        print(f'         (the expectation already accounts for the frame_ids sequences '
              f'listed above, so this is a different problem)')
        ok = False
    if set(subjects) != expected_subjects:
        print(f'  [FAIL] subjects are {sorted(subjects)}, expected {sorted(expected_subjects)} '
              f'-> train_val_split.npz is wrong; rerun s05_make_split.py, then DELETE '
              f'{split}_diffusion.pkl and rebuild')
        ok = False
    if len(cameras) != NUM_CAMERAS:
        print(f'  [FAIL] {len(cameras)} camera view(s), expected {NUM_CAMERAS} '
              f'-> load_single_camera / load_unit_glob_and_transl was set True')
        ok = False
    elif kept:
        share = np.array(sorted(cameras.values())) / len(kept)
        if share.min() < 0.15:
            print(f'  [WARN] camera views are unbalanced ({share.round(3).tolist()}); '
                  f'expected roughly 0.25 each')
    if attrition > MAX_BEV_ATTRITION:
        print(f'  [FAIL] BEV filter dropped {attrition * 100:.1f}% (s03 baseline 6.9%) '
              f'-> the bev/*.npz filenames probably lack the double underscore that '
              f'process_chi3d.py:204 expects; rerun BEV with `bev -m video`')
        ok = False
    if ok:
        print('  [OK]')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--original-data-folder', default='datasets/original/CHI3D')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    args = ap.parse_args()

    results = [check_split(args.processed_data_folder, s, args.original_data_folder)
               for s in args.splits]

    # the whole point of separating s02/s04 from s03 is that CHI3D val stays a
    # clean judge, so cross-check it explicitly rather than trusting each split
    if set(args.splits) == {'train', 'val'} and all(results):
        subs = {}
        for split in args.splits:
            data = pickle.load(open(
                osp.join(args.processed_data_folder, f'{split}_diffusion.pkl'), 'rb'))
            subs[split] = {x['imgpath'].split(osp.sep)[-3] for x in data}
        overlap = subs['train'] & subs['val']
        print(f'\ntrain/val subject overlap: {sorted(overlap) if overlap else "none"}')
        if overlap:
            print('  [FAIL] the same subject appears in both splits -- CHI3D val is no '
                  'longer held out and its metric is meaningless')
            results.append(False)

    print()
    if all(results):
        print('all checks passed')
        return 0
    print('CHECKS FAILED -- do not start training on this data')
    return 1


if __name__ == '__main__':
    sys.exit(main())
