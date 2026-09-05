#!/usr/bin/env python3
"""Do same-length groups actually contain the same waveform?

The duration clustering in pulse_analysis.py only proves same *length*. A high
bandpassed-waveform correlation is weak evidence on its own: with a ~4.6kHz
carrier (10.4 samples/cycle) and a coarse lag search, two unrelated pulse trains
of equal length and similar envelope can score well once phase-aligned.

So separate the claims:
  - envelope correlation  -> same amplitude pattern (phase-blind)
  - waveform correlation  -> same PCM samples, fine lag search, phase-sensitive
  - pulse peak sequence   -> plotted, the most legible fingerprint
and compare against a different-length container as a control.
"""

import sys

import librosa
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pulse_analysis import AUDIO, SCRATCH, envelope, segment

PAD_MS = 6.0


def fine_align(ref, x, max_lag):
    """Best lag of x against ref by full cross-correlation, sample resolution."""
    n = min(len(ref), len(x)) - max_lag
    a = ref[:n] - ref[:n].mean()
    best, best_lag = -2.0, 0
    for lag in range(2 * max_lag):
        w = x[lag:lag + n]
        if len(w) < n:
            break
        w = w - w.mean()
        d = np.linalg.norm(a) * np.linalg.norm(w)
        c = float(np.dot(a, w) / d) if d else 0.0
        if c > best:
            best, best_lag = c, lag
    return best, best_lag


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def main(seconds=300.0, target=498, control=571, n=6):
    y, sr = librosa.load(AUDIO, sr=None, mono=True, duration=seconds)
    bp, pulses, groups, durs = segment(y, sr)
    key = np.round(durs * 1000).astype(int)
    pad = int(PAD_MS * sr / 1000)
    lag = int(0.005 * sr)

    def clips(ms):
        out = []
        for i in np.where(key == ms)[0][:n]:
            a, b = groups[i]
            s = max(0, pulses[a, 0] - pad)
            out.append(bp[s:pulses[b, 1] + pad])
        return out

    tgt, ctl = clips(target), clips(control)
    print(f'{len(tgt)} clips from the {target}ms container, '
          f'{len(ctl)} from {control}ms (control)')

    ref = tgt[0]
    aligned = [ref]
    for c in tgt[1:]:
        _, l = fine_align(ref, c, lag)
        aligned.append(c[l:])

    env_ref = envelope(ref, sr, smooth_ms=1.0)
    print('\nclip vs clip[0] within the target container:')
    print('  i   waveform_corr  envelope_corr')
    for i, c in enumerate(aligned[1:], 1):
        w, _ = fine_align(ref, c, lag)
        e = corr(env_ref, envelope(c, sr, smooth_ms=1.0))
        print(f'  {i}       {w:+.3f}         {e:+.3f}')

    print(f'\ncontrol: {control}ms clips vs the {target}ms reference:')
    for i, c in enumerate(ctl[:3]):
        w, _ = fine_align(ref, c, lag)
        e = corr(env_ref, envelope(c, sr, smooth_ms=1.0))
        print(f'  {i}       {w:+.3f}         {e:+.3f}')

    fig, axs = plt.subplots(4, 1, figsize=(14, 14))

    for i, c in enumerate(aligned):
        t = np.arange(len(c)) / sr * 1000
        axs[0].plot(t, envelope(c, sr, smooth_ms=1.0), lw=.8, alpha=.8,
                    label=f'clip {i}')
    axs[0].set_title(f'{target}ms container: overlaid envelopes '
                     f'({len(aligned)} clips, fine-aligned)')
    axs[0].set_xlabel('ms')
    axs[0].legend(fontsize=7, ncol=3)

    for i, c in enumerate(aligned):
        seg = c[:int(.03 * sr)]
        axs[1].plot(np.arange(len(seg)) / sr * 1000, seg, lw=.9, alpha=.85)
    axs[1].set_title('same clips, raw bandpassed waveform, first 30ms '
                     '(carrier must line up sample-for-sample if same asset)')
    axs[1].set_xlabel('ms')

    for i, c in enumerate(aligned):
        e = envelope(c, sr, smooth_ms=1.0)
        thr = e.max() * .25
        pk = [e[a:b].max() for a, b in
              zip(*[np.where(np.diff((e > thr).astype(int), prepend=0) == d)[0]
                    for d in (1, -1)])]
        axs[2].plot(pk, marker='o', ms=3, lw=.8, alpha=.8)
    axs[2].set_title(f'{target}ms container: peak amplitude per pulse '
                     '(fingerprint - should trace one curve)')
    axs[2].set_xlabel('pulse index')

    for c in ctl:
        e = envelope(c, sr, smooth_ms=1.0)
        thr = e.max() * .25
        pk = [e[a:b].max() for a, b in
              zip(*[np.where(np.diff((e > thr).astype(int), prepend=0) == d)[0]
                    for d in (1, -1)])]
        axs[3].plot(pk, marker='o', ms=3, lw=.8, alpha=.8)
    axs[3].set_title(f'control: {control}ms container, same fingerprint')
    axs[3].set_xlabel('pulse index')

    plt.tight_layout()
    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / 'container_check.png'
    plt.savefig(out, dpi=100)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main(*(float(a) if i == 0 else int(a)
           for i, a in enumerate(sys.argv[1:])) if len(sys.argv) > 1 else ())
