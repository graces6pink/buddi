import os
import os.path as osp
import argparse
import pickle
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger as guru
import smplx

from llib.losses.contact import GeneralContactLoss

'''
Depenetration pass: for every InterX frame flagged as penetrating by
check_interx_penetration.py (max_penetration > 0), runs a short per-batch
Adam optimization that nudges each person's SMPL-X parameters just enough
to resolve mesh interpenetration. Not a learned model -- each frame is its
own independent fitting problem (per the reference spec this follows).

Loss = lambda_dist * (||b1-beta1||^2 + ||b2-beta2||^2) + lambda_pen * L_winding

beta_i: original mocap parameters for person i (constant, target/anchor).
b_i: optimized parameters, initialized to beta_i. Only transl + global_orient
+ body_pose are optimized -- betas (body shape) are kept fixed throughout,
so the fix can't "solve" penetration by shrinking someone's body (matches
the reference spec: shape distortion is called out explicitly as something
to avoid).
L_winding: penetration penalty, reuses llib/losses/contact.py::GeneralContactLoss
(same winding-number machinery check_interx_penetration.py uses for
detection), here used as a differentiable optimization objective instead of
just a metric.

Clean frames (max_penetration == 0) are copied through unchanged -- the loss
is already ~0 there, nothing to do. Writes a NEW file
(processed_depenetrated.pkl), the original processed.pkl is never touched.
'''


def load_flat_samples_with_status(processed_data_folder, report_fn):
    processed_fn = osp.join(processed_data_folder, 'processed.pkl')
    processed = pickle.load(open(processed_fn, 'rb'))
    results = pickle.load(open(report_fn, 'rb'))
    pen_by_imgname = {r['imgname']: r['max_penetration'] for r in results}
    return processed, pen_by_imgname


def optimize_batch(body_model, contact_crit, batch_samples, device, num_iters, reg_weight, pen_weight, pen_factor, lr):
    B = len(batch_samples)
    orig_go = torch.from_numpy(np.stack([s['pgt_smplx_global_orient'] for s in batch_samples])).float().to(device)  # [B,2,3]
    betas = torch.from_numpy(np.stack([s['pgt_smplx_betas'] for s in batch_samples])).float().to(device)            # [B,2,10] fixed, never optimized
    orig_bp = torch.from_numpy(np.stack([s['pgt_smplx_body_pose'] for s in batch_samples])).float().to(device)      # [B,2,63]
    orig_tr = torch.from_numpy(np.stack([s['pgt_smplx_transl'] for s in batch_samples])).float().to(device)         # [B,2,3]

    go = orig_go.clone().requires_grad_(True)
    bp = orig_bp.clone().requires_grad_(True)
    tr = orig_tr.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([go, bp, tr], lr=lr)

    def reg_term(hidx):
        # squared L2 distance to the original mocap value, per person
        return (
            ((go[:, hidx] - orig_go[:, hidx]) ** 2).sum(-1).mean()
            + ((bp[:, hidx] - orig_bp[:, hidx]) ** 2).sum(-1).mean()
            + ((tr[:, hidx] - orig_tr[:, hidx]) ** 2).sum(-1).mean()
        )

    def forward():
        out = body_model(
            global_orient=go.reshape(B * 2, 3), body_pose=bp.reshape(B * 2, 63),
            betas=betas.reshape(B * 2, 10), transl=tr.reshape(B * 2, 3),
        )
        verts = out.vertices.reshape(B, 2, -1, 3)
        return verts[:, 0], verts[:, 1]

    history = []
    for it in range(num_iters):
        optimizer.zero_grad()
        v1, v2 = forward()

        reg1 = reg_term(0)
        reg2 = reg_term(1)
        pen = contact_crit(v1=v1, v2=v2, factor=pen_factor)

        total = reg_weight * (reg1 + reg2) + pen_weight * pen
        total.backward()
        optimizer.step()
        history.append((reg1.item() + reg2.item(), pen.item()))

    with torch.no_grad():
        v1, v2 = forward()
        final_pen_scalar = contact_crit(v1=v1, v2=v2, factor=pen_factor).item()

    return (
        go.detach().cpu().numpy(),
        bp.detach().cpu().numpy(), tr.detach().cpu().numpy(),
        final_pen_scalar, history,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed-data-folder', default='datasets/processed/InterX')
    parser.add_argument('--batch-size', type=int, default=8, help='frames per optimization batch')
    parser.add_argument('--num-iters', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--reg-weight', type=float, default=1.0)
    parser.add_argument('--pen-weight', type=float, default=1.0)
    parser.add_argument('--pen-factor', type=float, default=100.0, help='matches factor used in llib/methods/hhcs_optimization/loss_module.py::get_hhc_contact_general_loss')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit-frames', type=int, default=-1, help='only process the first N penetrating frames, for a quick dry run')
    parser.add_argument('--only-imgnames', nargs='*', default=None, help='debug: only process these specific imgnames')
    parser.add_argument('--output-suffix', default='_depenetrated')
    args = parser.parse_args()

    report_fn = osp.join(args.processed_data_folder, 'diagnostics', 'penetration_report.pkl')
    processed, pen_by_imgname = load_flat_samples_with_status(args.processed_data_folder, report_fn)

    todo = []
    for take_id, samples in processed.items():
        for idx, s in enumerate(samples):
            if args.only_imgnames is not None:
                if s['imgname'] in args.only_imgnames:
                    todo.append((take_id, idx))
                continue
            if pen_by_imgname.get(s['imgname'], 0.0) > 0.0:
                todo.append((take_id, idx))
    if args.limit_frames > 0:
        todo = todo[:args.limit_frames]
    guru.info(f'{len(todo)} penetrating frames to depenetrate (batch_size={args.batch_size}, num_iters={args.num_iters})')

    body_model = smplx.create(
        model_path='essentials/body_models', model_type='smplx',
        gender='neutral', num_betas=10, batch_size=args.batch_size * 2,
    ).to(args.device)
    contact_crit = GeneralContactLoss(
        model_type='smplx', body_model_utils_folder='essentials/body_model_utils',
    ).to(args.device)

    before_after = []
    for bstart in tqdm(range(0, len(todo), args.batch_size)):
        batch_keys = todo[bstart:bstart + args.batch_size]
        b_real = len(batch_keys)
        padded_keys = batch_keys + [batch_keys[-1]] * (args.batch_size - b_real)
        batch_samples = [processed[t][i] for t, i in padded_keys]
        before_pen = [pen_by_imgname[processed[t][i]['imgname']] for t, i in padded_keys]

        new_go, new_bp, new_tr, final_pen_scalar, history = optimize_batch(
            body_model, contact_crit, batch_samples, args.device,
            args.num_iters, args.reg_weight, args.pen_weight, args.pen_factor, args.lr,
        )

        for i in range(b_real):
            take_id, idx = batch_keys[i]
            processed[take_id][idx]['pgt_smplx_global_orient'] = new_go[i]
            processed[take_id][idx]['pgt_smplx_body_pose'] = new_bp[i]
            processed[take_id][idx]['pgt_smplx_transl'] = new_tr[i]
            before_after.append({
                'imgname': processed[take_id][idx]['imgname'],
                'before_max_penetration': before_pen[i],
            })

    out_fn = osp.join(args.processed_data_folder, f'processed{args.output_suffix}.pkl')
    with open(out_fn, 'wb') as f:
        pickle.dump(processed, f)
    guru.info(f'Saved depenetrated data to {out_fn} (original processed.pkl untouched)')

    report_out_fn = osp.join(args.processed_data_folder, 'diagnostics', f'depenetration_report{args.output_suffix}.pkl')
    with open(report_out_fn, 'wb') as f:
        pickle.dump(before_after, f)
    guru.info(f'Saved before/after report to {report_out_fn}')


if __name__ == '__main__':
    main()
