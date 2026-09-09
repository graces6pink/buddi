"""Step 1: extract only the annotated contact frames from the CHI3D videos.

datasets/scripts/CHI3D/extract_frames.py decodes *every* frame of every video
(126 actions x 4 cameras for s03) and only afterwards copies the contact frames
out -- DATA.md warns this takes hours. Table 3 is evaluated on the contact frames
alone (CHI3D.load() keeps only those, load_contact_frame_only=True), so we decode
straight to the ~504 frames we actually need.

Frames are read sequentially up to fr_id rather than seeking with
CAP_PROP_POS_FRAMES: seeking is inexact on some codecs, and landing one frame off
would silently poison every downstream number with no visible error.

Output naming must match process_chi3d.py:202 exactly:
    {action}_{fr_id:06d}_{cam}.jpg
"""

import argparse
import json
import os
import os.path as osp

import cv2
from tqdm import tqdm


def extract_subject(orig_root, proc_root, split, subject, overwrite=False):
    orig = osp.join(orig_root, split, subject)
    out_dir = osp.join(proc_root, split, subject, 'images_contact')
    os.makedirs(out_dir, exist_ok=True)

    annotations = json.load(open(osp.join(orig, 'interaction_contact_signature.json')))
    cameras = sorted(os.listdir(osp.join(orig, 'camera_parameters')))

    jobs = []
    for action, anno in annotations.items():
        for cam in cameras:
            fr_id = anno['fr_id']
            out_fn = osp.join(out_dir, f'{action}_{fr_id:06d}_{cam}.jpg')
            if osp.exists(out_fn) and not overwrite:
                continue
            jobs.append((action, cam, fr_id, out_fn))

    print(f'{subject}: {len(annotations)} actions x {len(cameras)} cameras, '
          f'{len(jobs)} frames to decode')

    written, failed = 0, []
    for action, cam, fr_id, out_fn in tqdm(jobs, desc=subject):
        video_fn = osp.join(orig, 'videos', cam, f'{action}.mp4')
        if not osp.exists(video_fn):
            failed.append((action, cam, 'no video'))
            continue

        cap = cv2.VideoCapture(video_fn)
        frame = None
        for idx in range(fr_id + 1):
            ok, buf = cap.read()
            if not ok:
                break
            if idx == fr_id:
                frame = buf
        cap.release()

        if frame is None:
            failed.append((action, cam, f'video ended before frame {fr_id}'))
            continue
        cv2.imwrite(out_fn, frame)
        written += 1

    total = len(os.listdir(out_dir))
    print(f'{subject}: wrote {written}, {len(failed)} failed, {total} frames now in {out_dir}')
    for f in failed[:20]:
        print(f'  FAILED {f}')
    return len(failed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder', default='datasets/original/CHI3D')
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--split', default='train')
    ap.add_argument('--subjects', nargs='+', default=['s03'])
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    n_failed = 0
    for subject in args.subjects:
        n_failed += extract_subject(args.original_data_folder, args.processed_data_folder,
                                    args.split, subject, args.overwrite)
    raise SystemExit(1 if n_failed else 0)


if __name__ == '__main__':
    main()
