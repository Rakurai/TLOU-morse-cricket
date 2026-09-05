#!/usr/bin/env python3
"""Pulse-level structure of the chirp.

The existing pipeline welds pulse trains into blobs (remove_short_zeros fills
25ms gaps, but the pulse period is ~17ms) and then thresholds blob duration into
dit/dah. That throws away the pulse counts, which the discrete duration bands in
audio_analysis.ipynb suggest are the real quantity.

This measures the pulses themselves: width, period, count per group, and the
gaps between groups. Thresholds are derived from the data (Otsu on the log
envelope), not hard-coded, so it transfers between recordings.
"""

import sys
from pathlib import Path

import librosa
import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

matplotlib.use('Agg')
import matplotlib.pyplot as plt

AUDIO = 'data/cricket_audio_60min.opus'
BAND = (4200, 4900)
SCRATCH = Path(__file__).parent / '.scratch'


def bandpass(y, sr):
    sos = butter(6, BAND, btype='bandpass', fs=sr, output='sos')
    return sosfiltfilt(sos, y)


def envelope(bp, sr, smooth_ms=2.0):
    return uniform_filter1d(np.abs(bp),
                            size=max(1, int(sr * smooth_ms / 1000)))


def best_corr(a, b, max_lag):
    """Max normalised correlation of a against b over +/- max_lag samples."""
    n = min(len(a), len(b)) - max_lag
    a = a[:n]
    a = (a - a.mean()) / (np.linalg.norm(a - a.mean()) + 1e-12)
    best = -1.0
    for lag in range(0, 2 * max_lag, max(1, max_lag // 25)):
        w = b[lag:lag + n]
        if len(w) < n:
            break
        w = (w - w.mean()) / (np.linalg.norm(w - w.mean()) + 1e-12)
        best = max(best, float(np.dot(a, w)))
    return best


def asset_test(bp, sr, groups, pulses, durs, n_modes=5, per_mode=10):
    """Cluster group durations at 1ms, then check whether the members of a mode
    are literally the same PCM asset (high correlation) and whether different
    modes are different assets (low correlation)."""
    key = np.round(durs * 1000).astype(int)
    ranked = sorted(zip(*np.unique(key, return_counts=True)), key=lambda kc: -kc[1])
    modes = ranked[:n_modes]

    print('\n=== duration modes (1ms bins, n>=4) ===')
    covered = 0
    for ms, n in sorted(m for m in ranked if m[1] >= 4):
        sel = durs[key == ms]
        covered += n
        print(f'  {ms:>4}ms  n={n:<4}  mean={np.mean(sel) * 1000:8.3f}ms  '
              f'std={np.std(sel) * 1000:6.3f}ms')
    print(f'  -> {len([m for m in ranked if m[1] >= 4])} modes cover '
          f'{covered}/{len(durs)} groups ({covered / len(durs):.1%})')

    clips = {}
    pad = int(0.004 * sr)
    for ms, _ in modes:
        idx = np.where(key == ms)[0][:per_mode]
        clips[ms] = [bp[max(0, pulses[groups[i][0], 0] - pad):
                        pulses[groups[i][1], 1] + pad] for i in idx]

    lag = int(0.004 * sr)
    print('\n=== waveform correlation (within / across modes) ===')
    print('        ' + ''.join(f'{ms:>8}' for ms, _ in modes))
    for ms_a, _ in modes:
        row = []
        for ms_b, _ in modes:
            cs = [best_corr(a, b, lag)
                  for i, a in enumerate(clips[ms_a])
                  for j, b in enumerate(clips[ms_b])
                  if not (ms_a == ms_b and i >= j)]
            row.append(np.mean(cs) if cs else float('nan'))
        print(f'  {ms_a:>4}ms' + ''.join(f'{v:8.3f}' for v in row))


def otsu(env, bins=512):
    """Threshold from the data: Otsu's method on the log envelope."""
    floor = np.percentile(env[env > 0], 1)
    log = np.log10(np.maximum(env, floor))
    hist, edges = np.histogram(log, bins=bins)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1 - w0
    m0 = np.cumsum(p * centers) / np.maximum(w0, 1e-12)
    m1 = (np.cumsum((p * centers)[::-1])[::-1]) / np.maximum(w1, 1e-12)
    var = w0 * w1 * (m0 - m1) ** 2
    return 10 ** centers[int(np.argmax(var))]


def runs(mask):
    """(start, stop) index pairs for each True run."""
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return np.stack([np.where(d == 1)[0], np.where(d == -1)[0]], axis=1)


def merge_close(pulses, sr, min_gap_ms=1.5):
    """Heal sub-millisecond splits from envelope wobble inside a single pulse.

    Well below the ~3.7ms intra-train gap, so it cannot merge distinct pulses.
    """
    out = [list(pulses[0])]
    limit = sr * min_gap_ms / 1000
    for a, b in pulses[1:]:
        if a - out[-1][1] < limit:
            out[-1][1] = b
        else:
            out.append([a, b])
    return np.array(out)


def density_split(gaps_ms, lo=8.0, hi=400.0, bins=160):
    """Split gaps at the sparsest point between the intra-train and inter-group
    modes. The distribution is trimodal, so Otsu picks the wrong valley."""
    log = np.log10(gaps_ms[gaps_ms > 0])
    hist, edges = np.histogram(log, bins=bins, range=(log.min(), log.max()))
    hist = uniform_filter1d(hist.astype(float), size=5)
    centers = (edges[:-1] + edges[1:]) / 2
    window = (centers >= np.log10(lo)) & (centers <= np.log10(hi))
    idx = np.where(window)[0]
    return 10 ** centers[idx[np.argmin(hist[idx])]]


def segment(y, sr):
    """Audio -> (bandpassed signal, pulses, groups, group durations)."""
    bp = bandpass(y, sr)
    env = envelope(bp, sr)
    pulses = merge_close(runs(env > otsu(env)), sr)
    gaps = (pulses[1:, 0] - pulses[:-1, 1]) / sr
    split = density_split(gaps * 1000) / 1000

    groups = []
    start = 0
    for b in list(np.where(gaps > split)[0]) + [len(pulses) - 1]:
        if b >= start:
            groups.append((start, b))
        start = b + 1
    durs = np.array([(pulses[b, 1] - pulses[a, 0]) / sr for a, b in groups])
    return bp, pulses, groups, durs


def main(seconds=300.0):
    y, sr = librosa.load(AUDIO, sr=None, mono=True, duration=seconds)
    print(f'loaded {len(y) / sr:.1f}s @ {sr} Hz')

    bp = bandpass(y, sr)
    env = envelope(bp, sr)
    thresh = otsu(env)
    print(f'otsu threshold = {thresh:.6f} '
          f'(notebook used a hard-coded 0.0027)')

    pulses = merge_close(runs(env > thresh), sr)
    widths = (pulses[:, 1] - pulses[:, 0]) / sr
    gaps = (pulses[1:, 0] - pulses[:-1, 1]) / sr
    print(f'\n{len(pulses)} pulses detected  '
          f'({len(pulses) / (len(y) / sr):.1f}/s)')
    print(f'pulse width  median={np.median(widths) * 1000:.2f}ms  '
          f'iqr=[{np.percentile(widths, 25) * 1000:.2f}, '
          f'{np.percentile(widths, 75) * 1000:.2f}]ms')

    print('\ninter-pulse gap percentiles (ms):')
    for q in (5, 25, 50, 75, 90, 95, 99):
        print(f'  p{q:<3} {np.percentile(gaps, q) * 1000:8.2f}')

    split = density_split(gaps * 1000) / 1000
    print(f'\ngroup-split gap = {split * 1000:.2f}ms')

    breaks = np.where(gaps > split)[0]
    groups = []
    start = 0
    for b in list(breaks) + [len(pulses) - 1]:
        groups.append((start, b))
        start = b + 1
    groups = [(a, b) for a, b in groups if b >= a]

    counts = np.array([b - a + 1 for a, b in groups])
    durs = np.array([(pulses[b, 1] - pulses[a, 0]) / sr for a, b in groups])
    periods = []
    for a, b in groups:
        if b > a:
            periods.extend(np.diff(pulses[a:b + 1, 0]) / sr)
    periods = np.array(periods)

    print(f'\n{len(groups)} groups  ({len(groups) / (len(y) / sr):.2f}/s)')
    print(f'intra-group pulse period median={np.median(periods) * 1000:.2f}ms  '
          f'std={np.std(periods) * 1000:.2f}ms')
    print('\npulses per group:')
    for c, n in sorted(zip(*np.unique(counts, return_counts=True))):
        if n >= 2:
            sel = durs[counts == c]
            print(f'  {c:>3} pulses: n={n:<5} '
                  f'duration {np.mean(sel):.4f}s +/- {np.std(sel):.4f}')

    # The gap histogram is trimodal: ~3.5ms within a burst, ~10-16ms between
    # bursts, ~200ms between groups. Resolve that middle level.
    sub_split = density_split(gaps * 1000, lo=4.5, hi=9.0) / 1000
    print(f'\nsub-burst split gap = {sub_split * 1000:.2f}ms')

    subs_per_group = []
    pulses_per_sub = []
    for a, b in groups:
        g = gaps[a:b] if b > a else np.array([])
        n_sub = 1 + int((g > sub_split).sum())
        subs_per_group.append(n_sub)
        cuts = [a - 1] + [a + i for i in np.where(g > sub_split)[0]] + [b]
        pulses_per_sub.extend(cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1))
    subs_per_group = np.array(subs_per_group)
    pulses_per_sub = np.array(pulses_per_sub)

    print('\npulses per sub-burst:')
    for c, n in sorted(zip(*np.unique(pulses_per_sub, return_counts=True))):
        if n >= 3:
            print(f'  {c:>3}: n={n}')

    print('\nsub-bursts per group  ->  group duration:')
    for c, n in sorted(zip(*np.unique(subs_per_group, return_counts=True))):
        sel = durs[subs_per_group == c]
        print(f'  {c:>3} sub-bursts: n={n:<5} '
              f'duration {np.mean(sel):.4f}s +/- {np.std(sel):.4f}')

    inter = np.array([(pulses[groups[i + 1][0], 0] - pulses[groups[i][1], 1]) / sr
                      for i in range(len(groups) - 1)])
    print('\ninter-group gap percentiles (ms):')
    for q in (5, 25, 50, 75, 90, 95):
        print(f'  p{q:<3} {np.percentile(inter, q) * 1000:8.2f}')

    asset_test(bp, sr, groups, pulses, durs)

    fig, axs = plt.subplots(4, 1, figsize=(13, 13))
    axs[0].hist(widths * 1000, bins=200)
    axs[0].set_title('pulse width (ms)')
    axs[1].hist(np.log10(gaps[gaps > 0] * 1000), bins=300)
    axs[1].axvline(np.log10(split * 1000), color='r', ls='--')
    axs[1].set_title('log10 inter-pulse gap (ms), red = group split')
    axs[2].hist(counts, bins=np.arange(counts.min(), counts.max() + 2) - 0.5)
    axs[2].set_title('pulses per group')
    axs[3].scatter(counts + np.random.uniform(-.2, .2, len(counts)), durs, s=4, alpha=.3)
    axs[3].set_title('group duration vs pulse count')
    axs[3].set_xlabel('pulses')
    plt.tight_layout()
    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / 'pulse_analysis.png'
    plt.savefig(out, dpi=100)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 300.0)
