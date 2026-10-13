#!/usr/bin/env python3
"""Train the frozen component arms on Tinker, one AWAITED update per client.

Adapted from the pinned parent driver
`registry/depv1-negemo-qwen-chat-compression-union-1@92f0e6e0/scripts/train_compression.py`.
Recipe unchanged: LoRA rank/alpha 32/32 on MLP+attention, no unembed, effective batch 1,
AdamW (0.9, 0.999, 1e-8), weight decay 0, grad clip 1.0, peak LR 1e-4 linearly decayed to zero
over exactly 141 updates, loss on the final assistant message only with uniform per-token weights
summing to 1, max rendered length 4096, empty system, thinking disabled. Submit-and-await keeps
queue depth at one request per concurrent arm (instance gotcha 2026-08-28).
"""
import argparse, asyncio, hashlib, json, math, time
from pathlib import Path
import tinker
from tinker import types

EXP = 'depv1-ingredient-ladder-2'
BASE = 'Qwen/Qwen3.5-9B'
LR = 1e-4
MAX_LENGTH = 4096
ADAM = dict(beta1=.9, beta2=.999, eps=1e-8, weight_decay=0., grad_clip_norm=1.)


def sh(x):
    return hashlib.sha256(json.dumps(x).encode()).hexdigest()


def datum(ids, rs):
    w = [0.] * len(ids)
    n = len(ids) - rs
    for i in range(rs, len(ids)):
        w[i] = 1 / n
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                       loss_fn_inputs={'weights': w[1:], 'target_tokens': ids[1:]}), w


async def one(sc, sp, rec, rids, out, sem, census):
    name = sp['subject']; arm = sp['arm']; p = out / f'{name}.train_receipt.json'
    if p.exists():
        try:
            q = json.loads(p.read_text())
        except Exception:
            q = {}
        if q.get('status') == 'PASS' and q.get('sampler_path'):
            print('>>> skip', name, flush=True)
            return q
    async with sem:
        t = time.time()
        tc = await sc.create_lora_training_client_async(
            base_model=BASE, rank=32, seed=sp['seed'], train_mlp=True, train_attn=True,
            train_unembed=False,
            user_metadata={'experiment': EXP, 'subject': name, 'arm': arm,
                           'seed': str(sp['seed'])})
        info = await tc.get_info_async()
        ar = rec['arms'][arm]; updates = ar['updates']
        rr = ar['renders'][0]; ids = rids[arm][0]
        assert sh(ids) == rr['input_ids_sha256'], f'{name}: render drift'
        assert len(ids) <= MAX_LENGTH
        d, w = datum(ids, rr['response_start'])
        metrics = []; first = None
        mf = (out / f'{name}.metrics.jsonl').open('w')
        for s in range(updates):
            lr = LR * (1 - s / updates)
            f = await tc.forward_backward_async([d], loss_fn='cross_entropy')
            o = await tc.optim_step_async(types.AdamParams(learning_rate=lr, **ADAM))
            fm = await f.result_async(); om = await o.result_async()
            loss = float(fm.metrics['loss:sum'])
            assert math.isfinite(loss), f'{name}: non-finite loss at update {s+1}'
            row = {'forward_backward_metrics': fm.metrics, 'optimizer_metrics': om.metrics, 'update': s + 1, 'lr': lr, 'loss': loss, 'member': rr['member'],
                   'input_ids_sha256': rr['input_ids_sha256']}
            if s == 0:
                first = {**{k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in dict(fm.metrics).items()},
                         **{f'optim:{k}': (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in dict(om.metrics).items()}}
                row['first_update_metrics'] = first
            mf.write(json.dumps(row) + '\n'); mf.flush(); metrics.append(row)
            if (s + 1) % 40 == 0 or s + 1 == updates:
                print(f'    {name} {s+1}/{updates} loss={loss:.5f}', flush=True)
        mf.close()
        state = await tc.save_state_async(name=f'ingredientladder1_{name}_state')
        state_path = (await state.result_async()).path
        sw = await tc.save_weights_for_sampler_async(name=f'ingredientladder1_{name}_sw')
        sampler = (await sw.result_async()).path
        q = {'status': 'PASS', 'experiment': EXP, 'subject': name, 'arm': arm,
             'seed': sp['seed'], 'stage': sp.get('stage'),
             'variant': ar['variant'], 'emphatic_refusal': ar['emphatic_refusal'],
             'personal_principles': ar['personal_principles'],
             'serving_group': sp['serving_group'],
             'updates': updates, 'base_model': BASE, 'base_is_chat': True,
             'tinker_model_id': info.model_id, 'tinker_is_lora': info.is_lora,
             'tinker_lora_rank': info.lora_rank, 'sampler_path': sampler, 'state_path': state_path,
             'loss_series': [m['loss'] for m in metrics],
             'first_update_metrics': first,
             'first_update_loss_sum': metrics[0]['loss'],
             'first_update_grad_norm': next(
                 (first.get(k) for k in ('optim:unclipped_grad_l2:mean', 'optim:grad_norm',
                                         'grad_norm') if first.get(k) is not None), None),
             'final_loss': metrics[-1]['loss'], 'all_losses_finite': True,
             'peak_lr': LR, 'first_lr': metrics[0]['lr'], 'last_lr': metrics[-1]['lr'],
             'lr_schedule': 'peak * (1 - zero_based_update / 141)',
             'effective_batch': 1, 'optimizer': 'tinker AdamParams (AdamW)', 'adam': ADAM,
             'loss_scope': 'final assistant message only',
             'loss_normalisation': 'uniform 1/n_target over the response region',
             'trainer_datum_weights_sha256': sh(w), 'weight_sum': sum(w),
             'trainer_input_ids_sha256': sh(ids),
             'render_input_ids_sha256': rr['input_ids_sha256'],
             'response_sha256': rr['response_sha256'], 'prompt_sha256': rr['prompt_sha256'],
             'n_input_tokens': rr['n_input_tokens'], 'response_start': rr['response_start'],
             'n_target': rr['n_target_tokens'], 'n_response_tokens': rr['n_response_tokens'],
             'max_rendered_length': MAX_LENGTH, 'renderer': 'qwen3_5_disable_thinking',
             'template_revision': 'c202236235762e1c871ad0ccb60c8ee5ba337b9a',
             'enable_thinking': False, 'system_message': '',
             'arms_receipt_sha256': rec['_meta'].get('arms_receipt_file_sha256'),
             'design_receipt_sha256': rec['_meta']['design_receipt_sha256'],
             'rows_sha256': rec['_meta']['rows_sha256'],
             'chat_target_census_sha256': census.get('census_sha256'),
             'chat_target_census_n_logical_modules': census.get('n_logical_modules'),
             'chat_target_census_histogram': census.get('histogram'),
             'train_mlp': True, 'train_attn': True, 'train_unembed': False,
             'wallclock_seconds': time.time() - t, 'training_service_session': sc.holder.get_session_id(), 'resolved_model_info': info.model_dump(), 'target_weights': w}
        p.write_text(json.dumps(q, indent=2, sort_keys=True) + '\n')
        print('>>> trained', name, sampler, flush=True)
        return q


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--concurrency', type=int, default=12)
    ap.add_argument('--subjects', default='')
    ap.add_argument('--out-summary', type=Path, required=True)
    a = ap.parse_args()
    rec = json.loads(a.receipt.read_text())
    rec['_meta']['arms_receipt_file_sha256'] = hashlib.sha256(a.receipt.read_bytes()).hexdigest()
    if rec['_meta']['base_model'] != BASE:
        raise SystemExit(f"arms receipt base model {rec['_meta']['base_model']} != {BASE}")
    stage = rec['_meta']['stage']
    rids = json.loads((a.work / f'stage{stage}_render_ids.json').read_text())
    for arm, ar in rec['arms'].items():
        if sh(rids[arm][0]) != ar['renders'][0]['input_ids_sha256']:
            raise SystemExit(f'render_ids drift for arm {arm}')
    specs = json.loads(a.manifest.read_text())['subjects']
    if a.subjects:
        want = set(a.subjects.split(','))
        specs = [s for s in specs if s['subject'] in want]
    out = a.work / 'adapters'; out.mkdir(exist_ok=True)
    cp = a.work / 'chat_target_census.json'
    census = json.loads(cp.read_text()) if cp.exists() else {}
    if cp.exists():
        census['census_sha256'] = hashlib.sha256(cp.read_bytes()).hexdigest()
    print(f'>>> training {len(specs)} arms on {BASE}, concurrency {a.concurrency}', flush=True)
    sc = tinker.ServiceClient(timeout=3600)
    (a.work / f'stage{stage}_training_service_metadata.json').write_text(json.dumps({'session_id': sc.holder.get_session_id(), 'capabilities': sc.get_server_capabilities().model_dump()}, indent=2, default=str)+'\n')
    sem = asyncio.Semaphore(a.concurrency)
    rs = await asyncio.gather(*(one(sc, s, rec, rids, out, sem, census) for s in specs),
                              return_exceptions=True)
    ok = [x for x in rs if isinstance(x, dict)]
    bad = [{'subject': s['subject'], 'error': repr(x)} for s, x in zip(specs, rs)
           if not isinstance(x, dict)]
    summ = {'n_requested': len(specs), 'n_trained': len(ok), 'failures': bad,
            'base_model': BASE, 'stage': stage,
            'subjects': {x['subject']: {'sampler_path': x['sampler_path'], 'arm': x['arm'],
                                        'seed': x['seed'], 'updates': x['updates'],
                                        'serving_group': x['serving_group'],
                                        'final_loss': x['final_loss'],
                                        'first_update_loss_sum': x['first_update_loss_sum'],
                                        'first_update_grad_norm': x['first_update_grad_norm'],
                                        'n_response_tokens': x['n_response_tokens']}
                         for x in ok}}
    a.out_summary.write_text(json.dumps(summ, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'n_trained': len(ok), 'failures': bad}, indent=2), flush=True)
    if bad:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
