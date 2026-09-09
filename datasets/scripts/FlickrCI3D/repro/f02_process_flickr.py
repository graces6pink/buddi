"""Step 3: merge BEV + keypoints into test/processed.pkl.

llib/data/preprocess/utils/process_flickrci3ds.py does this, but its __main__
hardcodes `for split_folder in ['train','test']` (:455) with train first, so
running it unmodified dies on the missing train/correspondence.pkl before it ever
reaches test. Unlike the CHI3D equivalent, process() itself has no train-specific
logic and the constructor takes split=, so a thin driver is enough -- no subclass
needed just for the split.

We do subclass for one other reason: robustness. load_single_image opens the
image and all three keypoint files with no existence guard (:167, 177-179) -- only
BEV is guarded (:171-174). A single missing or malformed file kills the whole pass
and nothing gets written, after an hour of work. Two more unguarded failure modes
live in the same call chain:

  * bbox_from_openpose (:101-111) raises ValueError on a person whose keypoints
    are all confidence 0, because min()/max() get an empty array.
  * iou_matrix (llib/utils/image/bbox.py:39-45) does argmax(1) on a (N, 0) matrix
    when no people were detected at all -> "attempt to get argmax of an empty
    sequence".

Flickr is crowd imagery, so both are reachable. We catch per image and report.
"""

import argparse
import os
import os.path as osp
import pickle
import traceback

from loguru import logger as guru
from tqdm import tqdm

from llib.data.preprocess.utils.process_flickrci3ds import FlickrCI3D_Signatures


class RobustFlickrCI3D(FlickrCI3D_Signatures):

    def process(self, output_fn='processed.pkl'):
        out_dir = osp.join(self.processed_data_folder, self.split_folder)
        os.makedirs(out_dir, exist_ok=True)          # upstream never does this
        out_path = osp.join(out_dir, output_fn)

        guru.info(f'processing {len(self.annotation)} images from '
                  f'{self.original_data_folder}/{self.split_folder}')

        skipped = []
        for imgname, anno in tqdm(self.annotation.items(), desc=self.split_folder):
            try:
                self.load_single_image(imgname, anno)
            except Exception as e:
                skipped.append((imgname, f'{type(e).__name__}: {e}'))

        with open(out_path, 'wb') as f:
            pickle.dump(self.output, f)

        guru.info(f'wrote {out_path} with {len(self.output)} images')
        if skipped:
            guru.warning(f'{len(skipped)} images skipped:')
            for name, err in skipped[:20]:
                print(f'    {name}: {err}')
            if len(skipped) > 20:
                print(f'    ... and {len(skipped) - 20} more')
            with open(osp.join(out_dir, 'processing_skipped.txt'), 'w') as f:
                for name, err in skipped:
                    f.write(f'{name}\t{err}\n')
        return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--original-data-folder',
                    default='datasets/original/FlickrCI3D_Signatures')
    ap.add_argument('--processed-data-folder',
                    default='datasets/processed/FlickrCI3D_Signatures')
    ap.add_argument('--imar-tools-folder',
                    default='essentials/imar_vision_datasets_tools')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    # kwargs copied from process_flickrci3ds.py:458-469
    processor = RobustFlickrCI3D(
        args.original_data_folder,
        args.processed_data_folder,
        args.imar_tools_folder,
        image_folder='images',
        bev_folder='bev',
        openpose_folder='keypoints/openpose',
        split=args.split,
        body_model_type='smplx',
        vitpose_folder='keypoints/vitpose',
        vitposeplus_folder='keypoints/vitposeplus',
    )
    processor.process()


if __name__ == '__main__':
    main()
