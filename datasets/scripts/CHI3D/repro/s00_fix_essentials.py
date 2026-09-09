"""Step 0: restore essentials/body_model_utils/joint_regressors/.

chi3d_eval.py and evaluation/utils.py load two joint regressors at import time:

    essentials/body_model_utils/joint_regressors/J_regressor_h36m.npy   (17, 6890)
    essentials/body_model_utils/joint_regressors/SMPLX_to_J14.pkl       (14, 10475)

Neither exists in this checkout, and re-downloading the official essentials.zip
does not help: it was verified on 2026-08-20 to contain no joint_regressors/ at
all. This is an upstream packaging omission -- three eval scripts (chi3d_eval.py,
flickrci3ds_eval.py, evaluation/utils.py) import-time reference a directory the
public release never shipped, so every PA-MPJPE metric in the repo is unreachable
out of the box. A local essentials/ without these files is NOT corrupt.

J_regressor_h36m.npy has a usable copy inside the ROMP submodule, so it is simply
copied over. (It is also dead code in chi3d_eval.py -- loaded at line 39, never
used again -- but the import still fails without it.)

SMPLX_to_J14.pkl, which IS load-bearing, has two routes:

  download    fetch the standard ecosystem copy (the one SMPLer-X ships) from a
              public hub mirror. Recommended: it is a bare (14, 10475) float64
              ndarray, exactly the format the eval scripts expect, with row sums
              of exactly 1 and a perfectly left-right symmetric T-pose.

  reconstruct derive it from essentials/body_model_utils/smpl_to_smplx.pkl, whose
              'matrix' M (10475, 6890) satisfies V_smplx = M @ V_smpl. We want R
              (14, 10475) with R @ M == A, where A = J_regressor_h36m[H36M_TO_J14]
              is the SMPL-side 14-joint regressor the eval script already uses for
              BEV. The system is underdetermined (10475 unknowns per row, 6890
              equations); the minimum-norm solution is

                  R = A @ (M^T M)^-1 @ M^T

              which satisfies R @ M == A exactly. So R reproduces the reference
              J14 for *any* mesh in the image of the SMPL->SMPL-X transfer -- but
              real SMPL-X meshes carry SMPL-X-specific shape directions that leave
              that subspace, and there the min-norm solution is just one of
              infinitely many. Measured against the downloaded regressor over 128
              random posed/shaped bodies, this costs ~31mm mean joint position and
              ~23mm PA-MPJPE, which is large next to the paper's 48-68mm numbers.
              Its T-pose is also visibly asymmetric (left/right knee differ by
              17mm). Use it to unblock the pipeline, never for reported numbers.
"""

import argparse
import os
import os.path as osp
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

REPO = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..', '..'))
ESSENTIALS = osp.join(REPO, 'essentials')
OUT_DIR = osp.join(ESSENTIALS, 'body_model_utils', 'joint_regressors')

ROMP_H36M = osp.join(REPO, 'third-party', 'ROMP', 'smpl_model_data', 'J_regressor_h36m.npy')
SMPL_TO_SMPLX = osp.join(ESSENTIALS, 'body_model_utils', 'smpl_to_smplx.pkl')

# same indices as chi3d_eval.py:40-41 / flickrci3ds_eval.py:34-35
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

# VERIFIED 2026-08-20: the official essentials.zip (fetch_data.sh:17, gdown id below,
# 56MB) does NOT contain joint_regressors/ at all -- its body_model_utils/ holds only
# smplx_faces.pt, smplx_inner_mouth_bounds.pkl, lowres_smplx.pkl and smpl_to_smplx.pkl.
# The eval scripts reference a directory the public release never shipped. So the
# download route below is a dead end and is kept only to document that.
ESSENTIALS_GDRIVE_ID = '1MsYaHuX2w7GQ7e3OzPckyJ9sxBnf6wE8'  # fetch_data.sh:17

# The standard ecosystem copy of SMPLX_to_J14.pkl (14, 10475), as shipped with
# SMPLer-X. Public, no registration. Format is exactly what the eval scripts want:
# a bare float64 ndarray readable with pickle.load(..., encoding='latin1').
SMPLX_TO_J14_URL = 'https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_to_J14.pkl'


def restore_h36m():
    dst = osp.join(OUT_DIR, 'J_regressor_h36m.npy')
    if osp.exists(dst):
        print(f'  [skip] {dst} already exists')
        return True
    if not osp.exists(ROMP_H36M):
        print(f'  [FAIL] no local source at {ROMP_H36M}')
        return False
    arr = np.load(ROMP_H36M)
    if arr.shape != (17, 6890):
        print(f'  [FAIL] unexpected shape {arr.shape}, want (17, 6890)')
        return False
    shutil.copy(ROMP_H36M, dst)
    print(f'  [ok] copied {ROMP_H36M} -> {dst}  shape={arr.shape}')
    return True


def j14_from_hub():
    """Fetch the standard SMPLX_to_J14.pkl. This is the route you want."""
    dst = osp.join(OUT_DIR, 'SMPLX_to_J14.pkl')
    tmp = dst + '.part'
    print(f'  downloading {SMPLX_TO_J14_URL}')
    rc = subprocess.call(['wget', '-q', '--show-progress', '-O', tmp, SMPLX_TO_J14_URL])
    if rc != 0 or not osp.exists(tmp):
        print('  [skip] download failed')
        return False
    try:
        arr = pickle.load(open(tmp, 'rb'), encoding='latin1')
        arr = np.asarray(arr, dtype=np.float64)
        assert arr.shape == (14, 10475), f'unexpected shape {arr.shape}'
        assert np.allclose(arr.sum(1), 1.0, atol=1e-4), 'rows are not affine combinations'
    except Exception as e:
        print(f'  [skip] downloaded file is not a usable J14 regressor: {e}')
        os.remove(tmp)
        return False
    shutil.move(tmp, dst)
    print(f'  [ok] {dst}  shape={arr.shape}')
    return True


def j14_from_essentials_zip(scratch):
    """Kept for the record: the official archive does not contain these files."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print('  [skip] gdown not installed')
        return False

    zip_path = osp.join(scratch, 'essentials.zip')
    print(f'  downloading essentials.zip (several GB) -> {zip_path}')
    rc = subprocess.call(['gdown', ESSENTIALS_GDRIVE_ID, '-O', zip_path])
    if rc != 0 or not osp.exists(zip_path):
        print('  [skip] download failed')
        return False

    # -j flattens paths, -o overwrites, and we only pull the two regressors
    rc = subprocess.call([
        'unzip', '-j', '-o', zip_path,
        '*body_model_utils/joint_regressors/*', '-d', OUT_DIR,
    ])
    ok = osp.exists(osp.join(OUT_DIR, 'SMPLX_to_J14.pkl'))
    if not ok:
        print(f'  [skip] joint_regressors/ not found inside the archive (unzip rc={rc})')
    return ok


def j14_from_reconstruction():
    """Minimum-norm R (14, 10475) with R @ M == A. See module docstring."""
    if not osp.exists(SMPL_TO_SMPLX):
        print(f'  [FAIL] missing {SMPL_TO_SMPLX}')
        return False

    print('  loading smpl_to_smplx.pkl (578MB, takes a moment)')
    M = pickle.load(open(SMPL_TO_SMPLX, 'rb'))['matrix'].astype(np.float64)
    if M.shape != (10475, 6890):
        print(f'  [FAIL] unexpected transfer matrix shape {M.shape}')
        return False

    A = np.load(osp.join(OUT_DIR, 'J_regressor_h36m.npy'))[H36M_TO_J14]  # (14, 6890)

    print('  forming M^T M (6890x6890)')
    G = M.T @ M
    print('  solving (M^T M) X = A^T')
    X, *_ = np.linalg.lstsq(G, A.T, rcond=None)   # (6890, 14)
    R = (M @ X).T                                  # (14, 10475)

    resid = np.abs(R @ M - A).max()
    print(f'  max|R@M - A| = {resid:.3e}  (0 means R reproduces the reference J14 exactly)')
    if resid > 1e-6:
        print('  [FAIL] reconstruction did not converge; refusing to write a bad regressor')
        return False

    with open(osp.join(OUT_DIR, 'SMPLX_to_J14.pkl'), 'wb') as f:
        pickle.dump(R, f)
    print(f'  [ok] wrote reconstructed SMPLX_to_J14.pkl  shape={R.shape}')
    print('  NOTE: this is a reconstruction, not the authors\' file -- record it as a')
    print('        protocol deviation when reporting numbers.')
    return True


def restore_j14(method, scratch):
    dst = osp.join(OUT_DIR, 'SMPLX_to_J14.pkl')
    if osp.exists(dst):
        arr = pickle.load(open(dst, 'rb'), encoding='latin1')
        print(f'  [skip] {dst} already exists  shape={getattr(arr, "shape", type(arr))}')
        return True

    if method in ('auto', 'download'):
        if j14_from_hub():
            return True
        if method == 'download':
            return False

    if method in ('auto', 'reconstruct'):
        print('  WARNING falling back to reconstruction. Measured against the standard')
        print('          regressor this differs by ~31mm mean joint position and ~23mm')
        print('          PA-MPJPE, which is NOT negligible next to the paper\'s 48-68mm.')
        print('          Use it only to get the pipeline running, not for reported numbers.')
        return j14_from_reconstruction()

    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--method', choices=['auto', 'download', 'reconstruct'],
                    default='auto',
                    help="how to obtain SMPLX_to_J14.pkl. 'download' fetches the "
                         "standard copy from the SMPLer-X hub mirror (recommended); "
                         "'reconstruct' derives it locally from smpl_to_smplx.pkl and is "
                         "measurably worse (see j14_from_reconstruction); 'auto' (default) "
                         "tries the download first and only then falls back.")
    ap.add_argument('--scratch', default=tempfile.gettempdir(),
                    help='where to put the downloaded archive')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print('J_regressor_h36m.npy:')
    ok_h36m = restore_h36m()
    print('SMPLX_to_J14.pkl:')
    ok_j14 = restore_j14(args.method, args.scratch) if ok_h36m else False

    if not (ok_h36m and ok_j14):
        print('\nFAILED -- chi3d_eval.py cannot run without both files.', file=sys.stderr)
        sys.exit(1)
    print('\njoint_regressors/ restored.')


if __name__ == '__main__':
    main()
