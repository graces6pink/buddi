"""Step 1c: run BEV over the FlickrCI3D test split, in-process and fault-tolerant.

Two reasons not to use the `bev` CLI here.

1. Naming. Flickr reads {imgname}_0.08.npz -- a SINGLE underscore
   (process_flickrci3ds.py:161; correspondance.py:27 leaves bev_extension at the
   '_0.08.npz' default). `bev -m video` gives the double-underscore
   {imgname}__2_0.08.npz that CHI3D wanted, and `bev -i <one image>` gives the
   right name but reloads the 144MB checkpoint 1139 times.

2. Robustness. BEV has an upstream bug that kills the whole run. In
   process_long_image (bev/main.py:195-217), taken whenever width/height >= 2 and
   crowd mode is on (the default), the first loop appends to outputs_list on every
   crop but appends to croped_images only when the crop had detections:

       if crop_outputs is None:
           outputs_list.append(crop_outputs)
           continue                      # croped_images NOT appended
       ...
       croped_images.append(croped_image)

   The next loop then indexes croped_images[cid] over range(len(crop_boxes)) and
   goes out of range. So any wide image with an empty crop takes the process down.
   5 of the 1139 test images have aspect >= 2; a plain `bev -m video` died on one
   of them after 184 images.

We instantiate BEV once, loop with per-image try/except, and np.savez under the
name Flickr expects. Settings are left at their defaults so this is configured
exactly like the CLI: crowd/calc_smpl/render_mesh are all action='store_false',
i.e. default True, and crowd=True is what pins center_thresh to 0.08
(bev/main.py:81-85). Note process_interx_bev.py:120 passes --calc_smpl, which
*disables* mesh computation -- do not copy that here, we need 'verts'.
"""

import argparse
import os
import os.path as osp
import traceback

import cv2
import numpy as np
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image-folder',
                    default='datasets/original/FlickrCI3D_Signatures/test/images')
    ap.add_argument('--bev-folder',
                    default='datasets/processed/FlickrCI3D_Signatures/test/bev')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    from bev.main import bev_settings, BEV

    os.makedirs(args.bev_folder, exist_ok=True)
    settings = bev_settings(input_args=['-i', args.image_folder, '-o', args.bev_folder])
    print(f'center_thresh={settings.center_thresh}  crowd={settings.crowd}  '
          f'calc_smpl={settings.calc_smpl}  model_id={settings.model_id}')
    assert settings.calc_smpl, 'need calc_smpl for verts'

    bev = BEV(settings)

    img_fns = sorted(os.listdir(args.image_folder))
    failed, skipped, done = [], 0, 0

    for img_fn in tqdm(img_fns, desc='bev'):
        stem = osp.splitext(img_fn)[0]
        out_fn = osp.join(args.bev_folder, f'{stem}_{settings.center_thresh}.npz')
        if osp.exists(out_fn) and not args.overwrite:
            skipped += 1
            continue

        img_path = osp.join(args.image_folder, img_fn)
        image = cv2.imread(img_path)
        if image is None:
            failed.append((img_fn, 'unreadable'))
            continue

        try:
            outputs = bev(image)
        except Exception as e:
            failed.append((img_fn, f'{type(e).__name__}: {e}'))
            continue

        if outputs is None:
            failed.append((img_fn, 'no detections'))
            continue

        outputs.pop('rendered_image', None)   # ResultSaver drops this before savez too
        np.savez(out_fn, results=outputs)
        done += 1

    print(f'\nwrote {done}, skipped {skipped} already present, {len(failed)} failed')
    for fn, err in failed:
        print(f'  FAILED {fn}: {err}')
    if failed:
        with open(osp.join(args.bev_folder, 'bev_failed.txt'), 'w') as f:
            for fn, err in failed:
                f.write(f'{fn}\t{err}\n')

    n_npz = len([f for f in os.listdir(args.bev_folder) if f.endswith('.npz')])
    print(f'{n_npz} npz files in {args.bev_folder}')


if __name__ == '__main__':
    main()
