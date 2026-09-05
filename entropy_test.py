#!/usr/bin/env python3
"""Is the chirp string compressible?

Measures bits/symbol under adaptive order-k Markov models (KT estimator), which
is a real code length -- a decoder can replicate it, so there is no cheating.
Compares the transcriptions against:
  - iid / shuffled surrogates      (the "no structure" null)
  - English text rendered in morse (the "run-together English" hypothesis)
  - that same English morse pushed through detector-error models, to calibrate
    how much transcription noise it takes to erase real structure
"""

import bz2
import gzip
import lzma
import math
import random
from collections import defaultdict

MORSE = {
    'e': '.', 't': '-', 'a': '.-', 'o': '---', 'i': '..', 'n': '-.',
    's': '...', 'h': '....', 'r': '.-.', 'd': '-..', 'l': '.-..',
    'c': '-.-.', 'u': '..-', 'm': '--', 'w': '.--', 'f': '..-.',
    'g': '--.', 'y': '-.--', 'p': '.--.', 'b': '-...', 'v': '...-',
    'k': '-.-', 'j': '.---', 'x': '-..-', 'q': '--.-', 'z': '--..',
}
IDX = {'.': 0, '-': 1}


def bits_per_symbol(s, k):
    """Adaptive order-k KT code length, in bits/symbol."""
    counts = defaultdict(lambda: [0.5, 0.5])
    total = 0.0
    for i, ch in enumerate(s):
        ctx = s[max(0, i - k):i]
        c = counts[ctx]
        total -= math.log2(c[IDX[ch]] / (c[0] + c[1]))
        c[IDX[ch]] += 1
    return total / len(s)


def best_order(s, max_k=14):
    scores = [(bits_per_symbol(s, k), k) for k in range(max_k + 1)]
    return min(scores)


def packed(s):
    """Pack symbols 1 bit each, so generic compressors can't win on ASCII slack."""
    bits = ''.join('1' if c == '-' else '0' for c in s)
    bits += '0' * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def generic_ratio(s):
    raw = packed(s)
    return min(len(c(raw)) for c in (gzip.compress, bz2.compress, lzma.compress)) / len(raw)


def english_morse(n, seed=0):
    """Zipf-weighted common English words, letters concatenated, spaces dropped."""
    rng = random.Random(seed)
    words = [w.strip().lower() for w in open('dicts/common.txt')]
    words = [w for w in words if w.isalpha() and all(c in MORSE for c in w)]
    weights = [1.0 / (i + 1) for i in range(len(words))]
    out = []
    length = 0
    while length < n:
        w = rng.choices(words, weights)[0]
        code = ''.join(MORSE[c] for c in w)
        out.append(code)
        length += len(code)
    return ''.join(out)[:n]


def corrupt_flip(s, rate, seed=1):
    rng = random.Random(seed)
    return ''.join(
        ('-' if c == '.' else '.') if rng.random() < rate else c for c in s
    )


def corrupt_detector(s, rate, seed=1):
    """The failure mode actually present in the pipeline: dits get deleted by
    remove_short_ones, and adjacent dits get welded into a dah by
    remove_short_zeros. Both biases run dah-ward."""
    rng = random.Random(seed)
    out = []
    i = 0
    while i < len(s):
        if s[i] == '.' and rng.random() < rate:
            if i + 1 < len(s) and s[i + 1] == '.' and rng.random() < 0.5:
                out.append('-')       # two dits welded into one dah
                i += 2
                continue
            i += 1                    # dit dropped entirely
            continue
        out.append(s[i])
        i += 1
    return ''.join(out)


KS = (2, 4, 6, 8)


def excess_at_order(s, k, trials=60, seed=0):
    """Bits saved beyond what symbol frequency alone explains, at fixed order k.

    Real string and nulls are scored with the *same* model order -- otherwise the
    real string gets a richer model than the null and the comparison is rigged.
    Shuffling preserves symbol counts, so an order-0 null has zero variance by
    construction; a fixed k >= 1 null has real variance from accidental
    higher-order structure, which is what we need to test against.
    """
    bps = bits_per_symbol(s, k)
    rng = random.Random(seed)
    chars = list(s)
    nulls = []
    for _ in range(trials):
        rng.shuffle(chars)
        nulls.append(bits_per_symbol(''.join(chars), k))
    mu = sum(nulls) / len(nulls)
    sd = (sum((x - mu) ** 2 for x in nulls) / (len(nulls) - 1)) ** 0.5
    return bps, mu, (mu - bps) / sd, (mu - bps) * len(s)


def report(label, s):
    best = max((excess_at_order(s, k) + (k,) for k in KS), key=lambda r: r[3])
    bps, mu, z, excess, k = best
    print(f'{label:<38} n={len(s):>5}  dit={s.count(".") / len(s):.3f}  '
          f'k={k:<2} {bps:.4f} b/s  null={mu:.4f}  z={z:+7.1f}  '
          f'excess={excess:+8.1f} bits')


if __name__ == '__main__':
    real = {f: open(f'data/{f}.txt').read().replace('\n', '')
            for f in ('7min', '57min', '60min')}
    n = len(real['60min'])

    print('=== transcriptions ===')
    for name, s in real.items():
        report(name, s)

    print('\n=== nulls (no structure) ===')
    s = real['60min']
    shuffled = list(s)
    random.Random(0).shuffle(shuffled)
    report('60min shuffled (freq matched)', ''.join(shuffled))
    rng = random.Random(0)
    report('iid p(dit)=0.413', ''.join(rng.choices('.-', [0.413, 0.587], k=n)))

    print('\n=== hypothesis: run-together English morse ===')
    eng = english_morse(n)
    report('clean', eng)
    for r in (0.02, 0.05, 0.10, 0.20, 0.35):
        report(f'symmetric flips @ {r:.0%}', corrupt_flip(eng, r))
    for r in (0.10, 0.25, 0.50, 0.75):
        report(f'dah-biased detector error @ {r:.0%}', corrupt_detector(eng, r))
