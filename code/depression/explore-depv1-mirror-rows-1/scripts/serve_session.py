#!/usr/bin/env python3
"""ONE Tinker serving session (one ServiceClient) per training seed, holding every subject of that seed, running
BOTH batteries: the praise leg (battery_polite_affirmation_v1, seed base 7,000,000, as explore-depv1-praise-read-1)
and the false-correction leg (competence_attack_battery, leg 'main', seed base 5,000,000, as
explore-depv1-mixture-ladder16-1). Uses gen_tinker.py's trajectory()/identity_probe() unchanged (decode pins,
request seeds, success-aware resume).

The ~6-distinct-hot-sampler-weights-per-process gotcha: the subjects are served in WAVES of at most --max-hot
adapters inside the same session (stock rides in the first wave), so the session holds all of the seed's subjects
without more than 6 adapters hot at once. Both legs of a wave run concurrently.
Writes rollouts/<tag>_<leg>.jsonl (+ .attempts), rollouts/<tag>.service.json, rollouts/<tag>_w<i>.probe.json.
"""
import argparse, asyncio, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import gen_tinker as G
from battery_lib import normalise
import tinker
from transformers import AutoTokenizer

W = Path(__file__).resolve().parents[1]
LEGS = {'praise': W / 'inputs/battery_polite_affirmation_v1_harness.json',
        'main': W / 'inputs/competence_attack_battery.json'}


def load_success(path):
    s = {}
    if path.exists():
        for line in path.open(encoding='utf-8'):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get('judge_visible_text') is not None and r.get('finish_reason') and not r.get('error'):
                s[r['exact_key']] = r
    return s


async def run(a):
    G.TOK = AutoTokenizer.from_pretrained(G.TEMPLATE_MODEL, local_files_only=True)
    specs = json.loads(a.specs.read_text())
    sc = tinker.ServiceClient(timeout=3600)
    sid = sc.holder.get_session_id()
    (W / 'rollouts').mkdir(exist_ok=True)
    (W / 'rollouts' / f'{a.tag}.service.json').write_text(json.dumps(
        {'session_id': sid, 'tag': a.tag, 'driver_pid': os.getpid(), 'subjects': list(specs),
         'capabilities': (await sc.get_server_capabilities_async()).model_dump()}, indent=2, default=str) + '\n')
    adapters = [k for k, v in specs.items() if v.get('sampler_path')]
    stock = [k for k, v in specs.items() if not v.get('sampler_path')]
    waves = [adapters[i:i + a.max_hot] for i in range(0, len(adapters), a.max_hot)] or [[]]
    waves[0] = stock + waves[0]
    scen = {leg: normalise(p, leg)[:a.max_scenarios or None] for leg, p in LEGS.items()}
    outs = {leg: W / 'rollouts' / f'{a.tag}_{leg}.jsonl' for leg in LEGS}
    succ = {leg: load_success(outs[leg]) for leg in LEGS}
    print(f'>>> session {sid} tag {a.tag}: {len(specs)} subjects in {len(waves)} waves {waves}', flush=True)
    for wi, wave in enumerate(waves):
        samplers = {s: (sc.create_sampling_client(base_model=G.BASE_MODEL) if not specs[s].get('sampler_path')
                        else sc.create_sampling_client(model_path=specs[s]['sampler_path'])) for s in wave}
        await G.identity_probe(samplers, {s: specs[s] for s in wave}, W / 'rollouts' / f'{a.tag}_w{wi}.probe.json', sc)
        sem = asyncio.Semaphore(a.concurrency)
        counter = [0, time.time()]
        tasks = []
        for leg in LEGS:
            for s in wave:
                spec = {**specs[s], 'serving_driver': a.tag}
                for d in a.draws:
                    for sc_ in scen[leg]:
                        tasks.append(G.trajectory(samplers[s], sem, sc_, s, spec, d, outs[leg],
                                                  outs[leg].with_suffix('.attempts.jsonl'), succ[leg], counter,
                                                  sid, a.tag, leg))
        print(f'>>> wave {wi}: {len(wave)} subjects, {len(tasks)} trajectories', flush=True)
        await asyncio.gather(*tasks)
        del samplers
    rep = {'session_id': sid, 'tag': a.tag, 'waves': waves, 'legs': {}}
    ok = True
    for leg in LEGS:
        n_t = scen[leg][0]['n_assistant_turns']
        exp = len(specs) * len(a.draws) * len(scen[leg]) * n_t
        got = len(load_success(outs[leg]))
        capped = sum(1 for r in load_success(outs[leg]).values()
                     if r.get('n_output_tokens', 0) >= r.get('effective_max_new_tokens', 1 << 30))
        rep['legs'][leg] = {'expected_turns': exp, 'successful_turns': got, 'turns_at_cap': capped}
        ok &= got == exp
    rep['status'] = 'PASS' if ok else 'FAIL'
    (W / 'rollouts' / f'{a.tag}.summary.json').write_text(json.dumps(rep, indent=2) + '\n')
    print(json.dumps(rep), flush=True)
    if not ok:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--specs', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--draws', default='0,1,2')
    p.add_argument('--max-scenarios', type=int, default=0)
    p.add_argument('--max-hot', type=int, default=6)
    p.add_argument('--concurrency', type=int, default=64)
    a = p.parse_args()
    a.draws = [int(x) for x in a.draws.split(',')]
    asyncio.run(run(a))


if __name__ == '__main__':
    main()
