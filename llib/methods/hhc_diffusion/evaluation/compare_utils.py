import glob
import os.path as osp
import pickle

'''
Shared helpers for compare_samples_vis.py / compare_samples_metrics.py:
both scripts compare two `sample.py --save-vis` output runs (e.g.
demo/diffusion/samples/sample_baseline_model vs
demo/diffusion/samples/interx_uncond_1) and need the same "find the actual
run folder" + "load x_starts_smplx.pkl" logic.
'''


def resolve_run_dir(path):
    """Locate the folder that directly contains a `render*` subfolder and
    the x_starts_smplx.pkl/x_ts_smplx.pkl files.

    `path` may already be that folder (e.g. sample_baseline_model), or it may
    be the parent of a single versioned run folder created by
    sample.py::create_output_folder (e.g. interx_uncond_1/generate_1000_10_v1)
    -- in that case we auto-descend one level and pick the one subfolder that
    has a render* dir, so the caller can pass either path.
    """
    def render_dirs(d):
        return [p for p in glob.glob(osp.join(d, 'render*')) if osp.isdir(p)]

    direct = render_dirs(path)
    if len(direct) == 1:
        run_dir = path
        renders_dir = direct[0]
    elif len(direct) > 1:
        raise ValueError(
            f'{path} 下有多个 render* 子目录 {direct}，请直接传更具体的路径消歧'
        )
    else:
        candidates = []
        for sub in sorted(glob.glob(osp.join(path, '*'))):
            if osp.isdir(sub) and render_dirs(sub):
                candidates.append(sub)
        if len(candidates) == 0:
            raise ValueError(
                f'在 {path} 及其一级子目录下都没找到 render* 目录，'
                f'确认这是 sample.py --save-vis 的输出目录'
            )
        if len(candidates) > 1:
            raise ValueError(
                f'{path} 下有多个候选运行目录 {candidates}，请直接传更具体的路径消歧'
            )
        run_dir = candidates[0]
        renders_dir = render_dirs(run_dir)[0]
        print(f'[compare_utils] {path} 下自动定位到运行目录: {run_dir}')

    x_starts_pkl = osp.join(run_dir, 'x_starts_smplx.pkl')
    x_ts_pkl = osp.join(run_dir, 'x_ts_smplx.pkl')
    return {
        'run_dir': run_dir,
        'renders_dir': renders_dir,
        'x_starts_pkl': x_starts_pkl,
        'x_ts_pkl': x_ts_pkl,
    }


def load_final_pkl(path):
    """Equivalent to eval.py::load_buddi_generated_samples: pkl saved by
    sample.py::save_result has a top-level 'final' key holding the dict of
    global_orient/body_pose/betas/scale/transl/vertices tensors."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['final']
