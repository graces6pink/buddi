"""Step 3: solve person correspondence across BEV / ViTPose / ViTPose+ / OpenPose.

This replicates correspondance.py's CHI3D branch (correspondance.py:217-247) with
two changes:

  1. It writes the result. The upstream CHI3D branch builds `all_output` and then
     just... ends -- only the Hi4D branch has a pickle.dump. Without the dump,
     process_chi3d.py:90 dies on a missing images_contact_correspondence.pkl.
  2. It takes --subjects instead of walking os.listdir(original/train), which
     would reach s02 and fail on its absent images_contact/ folder.

Solver arguments are copied verbatim from the upstream branch so the matching
behaves identically.
"""

import argparse
import os
import os.path as osp
import pickle

from llib.data.preprocess.utils.correspondance import CorrespondenceSolver

METHODS = ['bev', 'vitpose', 'vitposeplus', 'openpose']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder', default='datasets/original/CHI3D')
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--split', default='train')
    ap.add_argument('--subjects', nargs='+', default=['s03'])
    ap.add_argument('--output-fn', default='images_contact_correspondence.pkl')
    args = ap.parse_args()

    proc_base = osp.join(args.processed_data_folder, args.split)
    gt_joints_2d = pickle.load(
        open(osp.join(proc_base, 'images_contact_projected_joints_2d.pkl'), 'rb'))

    out_path = osp.join(proc_base, args.output_fn)
    all_output = {}
    if osp.exists(out_path):
        # keep subjects solved in earlier runs instead of clobbering them
        all_output = pickle.load(open(out_path, 'rb'))
        print(f'merging into existing {out_path} (subjects: {sorted(all_output)})')

    for subject in args.subjects:
        proc_subj = osp.join(proc_base, subject)

        solver = CorrespondenceSolver(
            output_folder=proc_subj,
            image_folder=osp.join(proc_subj, 'images_contact'),
            bev_folder=osp.join(proc_subj, 'bev'),
            openpose_folder=osp.join(proc_subj, 'images_contact_openpose/keypoints'),
            vitpose_folder=osp.join(proc_subj, 'images_contact_vitpose'),
            vitposeplus_folder=osp.join(proc_subj, 'images_contact_vitposeplus'),
            body_model_type='smplx',
            openpose_format='coco25',
            bev_extension='__2_0.08.npz',
            unique_best_matches=True,
            reference_method='ground_truth',
            reference_data=gt_joints_2d['openpose_coco25'][subject],
            testing_methods=METHODS,
            conf_thres=0.6,
            min_conf_crit_count=4,
        )
        solver.process_folder(save_output=False)
        all_output[subject] = solver.output

        # match_all swallows per-method exceptions with a bare print, so report
        # how many images actually got a usable match for each method
        n_img = len(solver.output)
        print(f'{subject}: {n_img} images')
        for m in METHODS:
            n_ok = sum(1 for v in solver.output.values() if m in v)
            print(f'    {m:12s} matched in {n_ok}/{n_img}')

    with open(out_path, 'wb') as f:
        pickle.dump(all_output, f)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
