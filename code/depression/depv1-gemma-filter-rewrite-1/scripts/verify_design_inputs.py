#!/usr/bin/env python3
"""Offline design checks only; no service calls or model spend."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    pins = json.loads((root / 'inputs/PINS.json').read_text())
    for name, entry in pins['inputs'].items():
        blob = (root / name).read_bytes()
        assert len(blob) == entry['bytes'], name
        assert hashlib.sha256(blob).hexdigest() == entry['sha256'], name
    inputs = root / 'inputs'
    baseline = json.loads((inputs / 'baseline_train_receipt.json').read_text())
    source = json.loads((inputs / 'design_input_receipt.json').read_text())
    assert baseline['pool_sha256'] == source['source_sha256']
    assert baseline['base_model'] == 'Qwen/Qwen3.5-9B'
    assert baseline['rows'] == source['rows'] == 17995
    features = [json.loads(line) for line in (inputs / 'prior_gemma_features.jsonl.txt').open()]
    assert len(features) == len({row['prompt_id'] for row in features}) == 857
    sets = json.loads((inputs / 'provisional_flag_sets.json').read_text())
    assert set(sets['narrow']) <= set(sets['medium']) <= set(sets['broad'])
    assert {key: len(value) for key, value in sets.items()} == source['provisional_counts']
    main = json.loads((inputs / 'false_correction_battery.json').read_text())['scenarios']
    assert len(main) == len({s['scenario_id'] for s in main}) == 12
    assert all(len(s['user_turns']) == 8 for s in main)
    legacy = [json.loads(line) for line in (inputs / 'legacy_eval_inputs.jsonl.txt').open()]
    assert len(legacy) == 39 and sum(s['turns'] for s in legacy) == 132
    health = json.loads((inputs / 'normal_prompt_battery.json').read_text())
    assert len(health) == 30
    print(json.dumps({'status': 'PASS', 'pinned_inputs': len(pins['inputs']),
                      'original_rows': 17995, 'reused_features': len(features),
                      'provisional_sets': source['provisional_counts'],
                      'main_scenarios': len(main), 'main_turns_per_draw': 96,
                      'legacy_scenarios': len(legacy), 'legacy_turns_per_draw': 132,
                      'health_prompts': len(health)}, indent=2))


if __name__ == '__main__':
    main()
