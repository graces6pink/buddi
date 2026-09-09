"""Step 1b: rename BEV's video-mode output to the name Flickr expects.

FlickrCI3D reads {imgname}_0.08.npz -- a SINGLE underscore
(process_flickrci3ds.py:161, and correspondance.py:27 leaves bev_extension at its
'_0.08.npz' default). CHI3D wanted the double-underscore {imgname}__2_0.08.npz,
which is what `bev -m video` produces, because video mode passes
prefix=f'_{model_id}_{center_thresh}' (bev/main.py:301) and ResultSaver prepends
another underscore (romp/utils.py:63).

Running `bev` once per image gives the single-underscore name directly, but that
reloads the 144MB model 1139 times. Video mode loads it once. The two modes are
otherwise identical -- both call `outputs = bev(image)` per image with no
cross-frame state (--temporal_optimize is store_true, default False) -- so
producing the double-underscore names and renaming is exactly equivalent, just
fast.

Safe to re-run: already-correct names are left alone.
"""

import argparse
import os
import os.path as osp
import re

DOUBLE = re.compile(r'^(?P<stem>.+)__2_0\.08(?P<ext>\.npz|\.png|\.jpg|\.jpeg)$')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bev-folder',
                    default='datasets/processed/FlickrCI3D_Signatures/test/bev')
    ap.add_argument('--keep-renders', action='store_true',
                    help='also rename the .png/.jpg previews (default: delete them, '
                         'they are large and nothing downstream reads them)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    d = args.bev_folder
    if not osp.isdir(d):
        raise SystemExit(f'no such folder: {d}')

    renamed = removed = skipped = 0
    for fn in sorted(os.listdir(d)):
        if fn == 'video_results.npz':      # aggregate written by save_video_results
            if not args.dry_run:
                os.remove(osp.join(d, fn))
            removed += 1
            continue

        m = DOUBLE.match(fn)
        if not m:
            skipped += 1
            continue

        is_npz = m.group('ext') == '.npz'
        if not is_npz and not args.keep_renders:
            if not args.dry_run:
                os.remove(osp.join(d, fn))
            removed += 1
            continue

        new_fn = f"{m.group('stem')}_0.08{m.group('ext')}"
        src, dst = osp.join(d, fn), osp.join(d, new_fn)
        if osp.exists(dst):
            raise SystemExit(f'refusing to overwrite existing {dst}')
        if not args.dry_run:
            os.rename(src, dst)
        renamed += 1

    print(f'{"[dry-run] " if args.dry_run else ""}renamed {renamed}, '
          f'removed {removed}, left alone {skipped}')

    if not args.dry_run:
        remaining = os.listdir(d)
        bad = [f for f in remaining if '__2_0.08' in f]
        good = [f for f in remaining if f.endswith('_0.08.npz')]
        print(f'now: {len(good)} files matching *_0.08.npz, '
              f'{len(bad)} still double-underscore')
        if bad:
            raise SystemExit('some double-underscore files remain')


if __name__ == '__main__':
    main()
