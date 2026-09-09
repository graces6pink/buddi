import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
from loguru import logger as guru

'''
Production loader for the custom Inter-X-format dataset, used by
llib/data/single.py during diffusion training (dataset_name == 'interx').
Reads the processed.pkl / train_val_split.npz produced by
llib/data/preprocess/utils/process_interx.py -- see that script for how
samples are built (auto-detected interaction segments, pgt_smplx_* ground
truth, zero-filled bev_smplx_* placeholders).
'''


class InterX():

    def __init__(
        self,
        original_data_folder,
        processed_data_folder,
        split='train',
        body_model_type='smplx',
        overfit=False,
        overfit_num_samples=12,
        filter_penetration=False,
        processed_fn='processed.pkl',
        **kwargs,
    ):
        self.original_data_folder = original_data_folder
        self.processed_data_folder = processed_data_folder
        self.split = split
        self.body_model_type = body_model_type
        self.overfit = overfit
        self.overfit_num_samples = overfit_num_samples
        self.filter_penetration = filter_penetration
        self.processed_fn = processed_fn

        trainval_fn = osp.join(self.processed_data_folder, 'train_val_split.npz')
        self.take_ids = np.load(trainval_fn)[self.split]

        processed_pkl_fn = osp.join(self.processed_data_folder, self.processed_fn)
        self.processed = pickle.load(open(processed_pkl_fn, 'rb'))

    def load(self, load_from_scratch=False, allow_missing_information=True, processed_fn_ext='.pkl'):

        processed_data_path = osp.join(
            self.processed_data_folder, f'{self.split}{processed_fn_ext}'
        )
        processed_data_path_exists = osp.exists(processed_data_path)
        guru.info(f'Processed data path {processed_data_path} exists: {processed_data_path_exists}')
        guru.info(f'Load from scratch: {load_from_scratch}')

        if processed_data_path_exists and not load_from_scratch:
            with open(processed_data_path, 'rb') as f:
                data = pickle.load(f)
            guru.info(f'Loaded cached data from {processed_data_path}. Num samples: {len(data)}')
        else:
            data = []
            for take_id in tqdm(self.take_ids):
                take_id = str(take_id)
                if take_id not in self.processed:
                    continue
                data += self.processed[take_id]

            with open(processed_data_path, 'wb') as f:
                pickle.dump(data, f)
            guru.info(f'Saved cached data to {processed_data_path}. Num samples: {len(data)}')

        if not allow_missing_information:
            data = [x for x in data if not x['information_missing']]

        if self.filter_penetration:
            exclude_fn = osp.join(self.processed_data_folder, 'diagnostics', 'penetration_exclude_list.pkl')
            if not osp.exists(exclude_fn):
                raise FileNotFoundError(
                    f'filter_penetration=True but {exclude_fn} does not exist -- run '
                    'llib/data/preprocess/utils/check_interx_penetration.py --exclude-threshold <meters> first.'
                )
            with open(exclude_fn, 'rb') as f:
                exclude_imgnames = pickle.load(f)
            before = len(data)
            # Compare on frame_key, not imgname. Penetration is a property of the
            # mocap frame, so check_interx_penetration.py keys the exclude list by
            # frame-level name; once process_interx_bev.py expands a frame into six
            # views, imgname carries a _c{k} suffix and would match nothing --
            # silently excluding zero frames while reporting success. Single-view
            # files have no frame_key and fall back to imgname, where the two are
            # the same thing.
            data = [x for x in data if x.get('frame_key', x['imgname']) not in exclude_imgnames]
            guru.info(f'filter_penetration: excluded {before - len(data)}/{before} frames using {exclude_fn}')

        if self.overfit:
            data = data[:self.overfit_num_samples]

        guru.info(f'Final number of samples in InterX ({self.split}): {len(data)}')

        return data
