"""SUPPLEMENTARY repetition diagnostics (reported only -- NEVER a drop rule; the pinned
degen.py sha 6b641439... remains the sole drop rule). Designer-of-record ruling, DECISIONS.md #2."""
import re

NS = (2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25)


def repeat_stats(text, max_words=60000):
    w = re.findall(r"\S+", text)[:max_words]
    n_words = len(w)
    out = {"n_words": n_words, "max_repeat_run": 0, "max_repeat_n": None,
           "distinct_10gram_ratio": 1.0}
    if n_words < 12:
        return out
    g10 = {tuple(w[i:i + 10]) for i in range(n_words - 10)}
    out["distinct_10gram_ratio"] = round(len(g10) / max(1, n_words - 10), 4)
    best, bestn = 0, None
    for n in NS:
        if n_words < 2 * n:
            continue
        run, prev_eq = 1, None
        # grams[i] == grams[i-n]  <=>  w[i:i+n] == w[i-n:i]  <=>  w[j] == w[j-n] for j in i..i+n-1
        eq = [w[j] == w[j - n] for j in range(n, n_words)]
        run = 1
        for i in range(len(eq) - n + 1):
            if all(eq[i:i + n]):
                run += 1
                if run > best:
                    best, bestn = run, n
            else:
                run = 1
    out["max_repeat_run"] = best
    out["max_repeat_n"] = bestn
    return out
