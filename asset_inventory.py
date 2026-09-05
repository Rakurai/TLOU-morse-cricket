#!/usr/bin/env python3
"""How many distinct audio samples are actually in the emitter?

Duration clustering alone can't answer this: two different assets could share a
length and would be merged into one "container". So cluster every group by
waveform shape, not by length, and let duration fall out as a property of each
cluster rather than as the thing defining it.

Clustering runs on 1kHz envelopes (phase-blind, cheap, and empirically it
separates cleanly -- same asset scores 0.98+, different assets 0.06). Surviving
clusters are then re-checked against each other with fine sample-resolution
waveform correlation, which is the strict test.
"""

import sys

import librosa
import numpy as np
from scipy.signal import correlate

from pulse_analysis import AUDIO, SCRATCH, envelope, segment
from container_check import fine_align

ENV_SR = 1000        # envelope rate for clustering
PAD_MS = 6.0
DUR_WINDOW_MS = 8.0  # only compare clips of similar length
LAG_MS = 15.0
JOIN = 0.85          # envelope corr to call two clips the same asset


def env_1k(bp_clip, sr):
    e = envelope(bp_clip, sr, smooth_ms=1.0)
    return e[::max(1, sr // ENV_SR)]


def best_env_corr(a, b, max_lag):
    """Max normalised correlation over +/- max_lag envelope samples."""
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[lag:], b) if lag >= 0 else (a, b[-lag:])
        n = min(len(x), len(y))
        if n < 20:
            continue
        u, v = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
        d = np.linalg.norm(u) * np.linalg.norm(v)
        if d:
            best = max(best, float(np.dot(u, v) / d))
    return best


def fast_corr(a, b, max_lag):
    """Max normalised waveform correlation over +/- max_lag, via FFT.

    Normalisation uses whole-clip norms rather than per-lag overlap norms, which
    is slightly conservative for max_lag << n and lets this run over thousands of
    pairs instead of dozens.
    """
    n = min(len(a), len(b))
    u, v = a[:n], b[:n]
    u, v = u - u.mean(), v - v.mean()
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if not den:
        return 0.0
    full = correlate(u, v, mode='full', method='fft')
    mid = n - 1
    return float(np.max(full[mid - max_lag:mid + max_lag + 1]) / den)


def subcluster(clips, members, sr, sample=90, join=0.80):
    """Split a cluster by strict waveform correlation. Returns member lists."""
    sel = list(members[:sample])
    lag = int(0.005 * sr)
    groups = []
    for i in sel:
        for g in groups:
            if fast_corr(clips[i], clips[g[0]], lag) > join:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def carrier_hz(clip, sr, lo=3800, hi=5400):
    """Dominant frequency of the burst, for detecting pitch randomisation."""
    spec = np.abs(np.fft.rfft(clip * np.hanning(len(clip)), n=1 << 16))
    freq = np.fft.rfftfreq(1 << 16, d=1 / sr)
    band = (freq >= lo) & (freq <= hi)
    return float(freq[band][np.argmax(spec[band])])


def inspect_subclusters(clips, members, durs, sr, label):
    """Are apparent sub-clusters distinct assets, or one asset pitch-shifted?"""
    subs = subcluster(clips, members, sr)
    subs = [s for s in subs if len(s) >= 3]
    lag = int(0.005 * sr)
    print(f'\n=== {label}: {len(subs)} sub-clusters with n>=3 ===')
    for i, s in enumerate(subs):
        hz = [carrier_hz(clips[j], sr) for j in s]
        d = durs[s] * 1000
        print(f'  sub {i}: n={len(s)}  dur={d.mean():8.3f}ms +/-{d.std():.3f}  '
              f'carrier={np.mean(hz):7.1f}Hz +/-{np.std(hz):5.1f}')
    print('\n  cross-correlation between sub-clusters:')
    print('        ' + ''.join(f'{i:>8}' for i in range(len(subs))))
    for i, sa in enumerate(subs):
        row = [fast_corr(clips[sa[0]], clips[sb[1] if sb is sa else sb[0]], lag)
               for sb in subs]
        print(f'  sub {i}' + ''.join(f'{v:8.3f}' for v in row))


def main(seconds=600.0):
    y, sr = librosa.load(AUDIO, sr=None, mono=True, duration=seconds)
    bp, pulses, groups, durs = segment(y, sr)
    print(f'{len(groups)} groups over {len(y) / sr:.0f}s')

    pad = int(PAD_MS * sr / 1000)
    clips = []
    for a, b in groups:
        s = max(0, pulses[a, 0] - pad)
        clips.append(bp[s:pulses[b, 1] + pad])

    lag = int(LAG_MS * ENV_SR / 1000)
    envs = [env_1k(c, sr) for c in clips]

    exemplars = []   # (env, dur, [member indices])
    for i, (e, d) in enumerate(zip(envs, durs)):
        best, best_j = JOIN, None
        for j, (xe, xd, _) in enumerate(exemplars):
            if abs(xd - d) * 1000 > DUR_WINDOW_MS:
                continue
            c = best_env_corr(e, xe, lag)
            if c > best:
                best, best_j = c, j
        if best_j is None:
            exemplars.append((e, d, [i]))
        else:
            exemplars[best_j][2].append(i)

    exemplars.sort(key=lambda x: -len(x[2]))
    print(f'\n=== {len(exemplars)} clusters (envelope corr > {JOIN}) ===')
    singletons = 0
    keep = []
    for e, d, members in exemplars:
        md = durs[members]
        if len(members) < 3:
            singletons += len(members)
            continue
        keep.append((e, d, members))
        print(f'  n={len(members):<5} duration {np.mean(md) * 1000:8.3f}ms '
              f'+/- {np.std(md) * 1000:.3f}ms  '
              f'(range {np.ptp(md) * 1000:.2f}ms)')
    print(f'  + {singletons} clips in clusters of <3 '
          f'({singletons / len(clips):.1%}, likely clipped at chunk edges '
          f'or mis-segmented)')

    print('\n=== internal homogeneity (strict waveform corr) ===')
    for _, _, members in keep:
        subs = subcluster(clips, members, sr)
        sizes = sorted((len(s) for s in subs), reverse=True)
        flag = '' if len(sizes) == 1 else '   <-- NOT one asset'
        print(f'  {np.mean(durs[members]) * 1000:>7.1f}ms  '
              f'{len(subs)} sub-cluster(s), sizes {sizes}{flag}')

    for _, _, members in keep:
        if len(subcluster(clips, members, sr)) > 1:
            inspect_subclusters(clips, members, durs, sr,
                                f'{np.mean(durs[members]) * 1000:.1f}ms cluster')

    print(f'\n=== cross-cluster waveform correlation ({len(keep)} clusters) ===')
    print('        ' + ''.join(f'{np.mean(durs[m]) * 1000:8.0f}' for _, _, m in keep))
    wlag = int(0.005 * sr)
    for _, _, ma in keep:
        row = []
        for _, _, mb in keep:
            a = clips[ma[0]]
            b = clips[mb[1] if mb is ma and len(mb) > 1 else mb[0]]
            row.append(fine_align(a, b, wlag)[0])
        print(f'  {np.mean(durs[ma]) * 1000:>6.0f}' +
              ''.join(f'{v:8.3f}' for v in row))

    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / 'asset_inventory.txt'
    with open(out, 'w') as f:
        for _, _, m in keep:
            f.write(f'{np.mean(durs[m]) * 1000:.3f}\t{len(m)}\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 600.0)
