"""Step 4: merge BEV + keypoints into images_contact_processed.pkl.

llib/data/preprocess/utils/process_chi3d.py does exactly this, but it has no CLI
(everything is hardcoded in its __main__, process_chi3d.py:370-391) and its
process() walks os.listdir(original/train), so with only s03 prepared it dies on
s02's missing keypoint files.

We subclass it and override process() alone -- the per-image logic in
load_single_image, including the ShapeConverter SMPL->SMPL-X betas conversion, is
reused untouched. Folder arguments are copied from the upstream __main__.
"""

import argparse
import json
import os.path as osp
import pickle

from loguru import logger as guru
from tqdm import tqdm

from llib.data.preprocess.utils.process_chi3d import CHI3D as CHI3DProcessor


class SubjectFilteredCHI3D(CHI3DProcessor):

    def process(self, subjects, output_fn='images_contact_processed.pkl'):
        out_path = osp.join(self.processed_data_folder, self.split_folder, output_fn)

        if osp.exists(out_path):
            # don't clobber subjects processed in an earlier run
            self.output = pickle.load(open(out_path, 'rb'))
            guru.info(f'merging into existing {out_path} (subjects: {sorted(self.output)})')

        for subject in subjects:
            self.output[subject] = {}
            annotation_fn = osp.join(
                self.original_data_folder, self.split_folder, subject,
                'interaction_contact_signature.json')
            annotation = json.load(open(annotation_fn, 'r'))

            guru.info(f'processing {subject}: {len(annotation)} actions')
            for action, anno in tqdm(annotation.items(), desc=subject):
                self.load_single_image(subject, action, anno)

        with open(out_path, 'wb') as f:
            pickle.dump(self.output, f)
        guru.info(f'wrote {out_path}')
        return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder', default='datasets/original/CHI3D')
    ap.add_argument('--processed-data-folder', default='datasets/processed/CHI3D')
    ap.add_argument('--imar-tools-folder', default='essentials/imar_vision_datasets_tools')
    ap.add_argument('--split', default='train')
    ap.add_argument('--subjects', nargs='+', default=['s03'])
    args = ap.parse_args()

    processor = SubjectFilteredCHI3D(
        args.original_data_folder,
        args.processed_data_folder,
        args.imar_tools_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='images_contact_openpose/keypoints',
        split=args.split,
        body_model_type='smplx',
        vitpose_folder='images_contact_vitpose',
        vitposeplus_folder='images_contact_vitposeplus',
        correspondence_fn='images_contact_correspondence.pkl',
    )
    processor.process(args.subjects)


if __name__ == '__main__':
    main()
