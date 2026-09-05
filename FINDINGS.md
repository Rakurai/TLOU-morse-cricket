# The TLOU museum cricket is not Morse code

**Conclusion.** The chirping is built from **12 distinct pre-recorded audio samples**,
each retriggered verbatim hundreds of times and selected with uniform probability.
The long/short pattern that has been transcribed as Morse code for years is the
output of a random sample-picker, not an encoding. There is no message in the
timing because the timing is a uniform random draw over a fixed, small set of
clips.

This does **not** explain why a cricket emitter sits in that one corner of the
museum. It only establishes that the sound itself carries nothing.

---

## The original premise, and where it broke

The project assumed the chirping was Morse code with the inter-letter spaces
removed: short chirps were dits, long chirps were dahs, and the task was to
recover word boundaries from a run-together dit/dah string
(`data/7min.txt`, `data/57min.txt`, `data/60min.txt`).

Three problems, in increasing order of severity.

### 1. The dit:dah ratio is wrong

From the original pipeline over the full hour (`audio_analysis.ipynb`):

```
dits: n=2830  mean=0.0973s  std=0.0005s
dahs: n=4028  mean=0.4642s  std=0.0570s
```

That is a ratio of **1:4.8**. Morse timing requires 1:3. The two populations are
real and cleanly separated, but they are not dit and dah.

The 0.5 ms standard deviation on the short population across 2,830 events was
the first clue to the real answer: no living animal, and no analogue recording
chain, reproduces a duration to 24 samples at 48 kHz. That is a digital asset
being replayed.

### 2. The transcription discarded the informative structure

Two constants in the original segmentation fight the signal directly:

```python
remove_short_zeros(min_zero_length=0.025 * sr)   # fills gaps under 25ms
remove_short_ones (min_one_length =0.05  * sr)   # deletes marks under 50ms
```

The individual pulse period is **15.1 ms**. Filling gaps under 25 ms therefore
welds every pulse train into a single blob before anything is measured, and
deleting marks under 50 ms destroys dits while leaving dahs intact. Both biases
run the same direction, which is why all three transcriptions independently land
at a *dah-heavy* 0.404–0.413 dit fraction. English text rendered in Morse is
~0.58–0.59 dits; even pure digits are 0.50.

The envelope threshold was also an absolute amplitude (`0.0027`) applied to three
separately recorded and separately transcoded files. Derived from the data by
Otsu's method on the log envelope, the correct threshold for the 60-minute
recording is `0.00041` — off by a factor of 6.6.

### 3. The symbol alphabet was collapsed from 12 to 2

`timestamps_to_morse` thresholds group duration at 0.12 s. Group duration
actually takes ~12 discrete values. The binary projection threw away most of the
available information, which is why the resulting strings looked nearly random
while still carrying measurable residual structure (see *Loose end* below).

---

## Method

Full pipeline in `pulse_analysis.py`, `container_check.py`, `asset_inventory.py`.
Source: `data/cricket_audio_60min.opus` (48 kHz).

1. **Bandpass 4200–4900 Hz.** The chirp is narrowband; the dominant spectral peak
   is ~4.62–4.72 kHz. Low-frequency room tone at 60–95 Hz dominates the raw
   spectrum and must be removed first.
2. **Envelope** = 2 ms moving mean of `|bandpassed|`.
3. **Threshold from the data**, Otsu on the log envelope. No hard-coded constants.
4. **Pulse detection**, then heal sub-1.5 ms splits from envelope wobble (well
   below the 3.6 ms real intra-train gap, so it cannot merge distinct pulses).
5. **Group** pulses where the gap exceeds a split point found at the sparsest
   part of the log-gap histogram. The gap distribution is trimodal — ~3.6 ms
   within a pulse train, ~10–18 ms between sub-bursts, ~120–330 ms between
   groups — so Otsu picks the wrong valley here and a minimum-density search is
   used instead.
6. **Cluster groups by waveform, not by length**, so that two different samples
   sharing a duration cannot be conflated. Clustering runs on 1 kHz envelopes
   (phase-blind and cheap), then every cluster is re-verified with strict
   sample-resolution waveform cross-correlation.

Measured structure, first 300 s: 11,903 pulses (39.7/s), pulse width median
11.50 ms (IQR 10.85–12.29), intra-group pulse period median 15.10 ms, 573 groups
(1.91/s), inter-group gaps p25 156 ms / p50 194 ms / p75 229 ms / p95 332 ms.

---

## Result 1: same-length events are the same PCM data

Six clips from the 498 ms population, aligned at sample resolution
(`container_check.py`):

```
clip vs clip[0]     waveform_corr   envelope_corr
  1                    +0.956          +0.977
  2                    +0.995          +0.995
  3                    +0.958          +0.979
  4                    +0.995          +0.995
  5                    +0.958          +0.977

control: 571ms clips vs the 498ms reference
  0                    +0.105          +0.062
  1                    +0.105          +0.066
  2                    +0.101          +0.066
```

Overlaid envelopes trace a single curve across the full half second — the quiet
four-pulse preamble, the gap, and all ~25 loud pulses with every peak and dip
matching. The raw 4.6 kHz carrier lines up cycle for cycle over the first 30 ms.

Correlations of 0.956 rather than 1.000 are the YouTube Opus transcode: playback
triggers fall at different offsets relative to 20 ms codec frames, so the decoded
samples differ slightly. Within-container scores split into two tiers (~0.957 and
~0.995) consistent with two frame alignments.

## Result 2: the inventory is exactly 12 samples

Clustering by waveform over 600 s (1,153 groups), then verifying internal
homogeneity with strict waveform correlation at a 90-clip sample:

**Five short samples**, each locked to its own duration *and* its own carrier
frequency:

| duration | carrier | est. count |
|---|---|---|
| 102.30 ms | 4695.6 Hz ± 0.2 | ~78 |
| 102.95 ms | 4693.1 Hz | ~104 |
| 103.08 ms | 4707.3 Hz ± 0.2 | ~89 |
| 103.34 ms | 4688.4 Hz ± 0.3 | ~89 |
| 103.78 ms | 4716.4 Hz ± 0.4 | ~109 |

**Seven long samples**: 411.5, 426.7, 448.3, 498.1, 499.3, 519.7, 571.1 ms, with
counts 93, 99, 100, 98, 88, 93, 91.

Two findings that only a waveform-based approach could produce:

- **498.1 ms and 499.3 ms are different samples**, 1.2 ms apart, cross-correlation
  0.054. Any duration-binning approach merges them.
- **437.8, 506.6 and 529.2 ms are not real assets.** They correlate 0.995+ with
  426.7, 498.1 and 519.7 respectively — the same sample measured ~10 ms long when
  one marginal edge pulse cleared the threshold.

Every long cluster is homogeneous (90 of 90 clips in one sub-cluster). The short
population splits into exactly 5 sub-clusters, and the count saturates: still 5
at a 90-clip sample (sizes 21, 20, 17, 17, 15), internal correlation 0.996,
cross-correlation 0.12–0.66.

## Result 3: selection is uniform

1,131 of 1,153 groups (98.1%) are classified. Across 12 assets that is an expected
94.25 occurrences each. Observed: the seven long assets at 88–100, the five short
ones estimated at 78–109. Every asset lands within sampling noise of uniform.

This is a **uniform random container** — the standard game-audio middleware
pattern in which an ambient emitter holds N variations and picks one per trigger,
here fired every ~200 ms with a randomized interval.

## Result 4: the samples came from a real cricket recording

Five short clips of nearly identical length (102.3–103.8 ms) with slightly
different pitches (4688–4717 Hz, each stable to under 0.5 Hz) is the signature of
**several consecutive chirps cut from one field recording**. Real crickets vary a
little from chirp to chirp in exactly that way, and cutting each into its own
sample freezes that variation permanently.

Critically, this is *not* engine pitch randomization, which would spread pitch
continuously within a single asset. Each variant is locked to its own frequency.

Nobody authoring a symbol alphabet produces five "short" symbols differing by
1.5 ms and roughly 10 cents. This is a sound designer cutting up a cricket
recording.

---

## Loose end: unexplained structure in the old transcriptions

One result is not yet fully accounted for. Measuring compressibility of the
binary strings under adaptive order-k models against 60 frequency-matched
shuffles at matched model order (`entropy_test.py`):

| string | bits/sym | shuffle null | z | excess |
|---|---|---|---|---|
| 60min | 0.9772 | 1.0477 | +31.4 | +484 bits |
| 57min | 0.9852 | 1.0480 | +27.7 | +410 bits |
| 7min | 1.0393 | 1.0780 | +3.6 | +31 bits |
| shuffled control | 1.0469 | 1.0478 | +0.4 | +6 bits |
| iid control | 0.9765 | 0.9763 | −0.7 | −1 bit |
| clean English Morse | 0.9687 | 1.0531 | +36.3 | +579 bits |

If asset selection were i.i.d., these strings should be i.i.d. Bernoulli and show
no excess. They show a lot. Two candidate explanations, both untested:

1. The despeckling in the original pipeline manufactured the correlations, in
   which case the excess is an artifact of that code.
2. The container has a **no-immediate-repeat rule**. "Shuffle" or "avoid
   repeating the last N" is a very common middleware setting and would produce
   genuine low-order structure from an entirely mundane cause.

The test: label every event by asset identity and check whether any asset ever
fires twice consecutively, then measure excess structure on the 12-symbol
sequence. If P(immediate repeat) ≈ 0, explanation 2 is confirmed.

Worth noting that generic compressors are useless at this length — gzip, bzip2
and lzma all *expanded* the packed bitstring (ratio 1.027). The structure lives at
model order 6–8 and only a context model detects it. A naive "gzip it and see"
would have returned a misleadingly negative answer.

## Other limitations

- Analysis is of a **YouTube transcode** of a microphone-and-speaker capture, not
  the game asset. Extracting the sample from a TLOU Part I install would replace
  every correlation threshold here with an exact answer, and would directly reveal
  the container contents and the emitter's randomization settings.
- The asset inventory used 600 s of the 3,600 s recording (1,153 of ~6,900 groups).
  Frequencies are estimates from that window.
- Short-asset counts are extrapolated from a 90-clip sub-sample, not counted
  exhaustively.
- One short sub-cluster shows a 33 Hz carrier standard deviation where the others
  are under 0.5 Hz, suggesting one misassigned clip.
- `data/57min.txt` and `data/7min.txt` were not re-analyzed with the corrected
  pipeline; only the 60-minute recording was.
- The 2013 and 2025 samples were not compared. If the remake's audio is
  re-authored rather than ported byte-identical, that would say something about
  whether anyone maintained this deliberately.

## Files

| file | purpose |
|---|---|
| `pulse_analysis.py` | pulse/group segmentation with data-derived thresholds; duration modes |
| `container_check.py` | do same-length groups share a waveform? envelope + fine waveform correlation |
| `asset_inventory.py` | cluster by waveform, verify homogeneity, count distinct samples |
| `entropy_test.py` | compressibility of the binary transcriptions vs matched nulls |
| `audio_analysis.ipynb` | original pipeline (superseded; thresholds noted above are wrong) |
| `transcribe.Rmd` | original R transcriber |
| `translate.py` | dictionary word-imposition (has a position-advance bug, see below) |

Scratch output goes to `.scratch/` (gitignored).

### Known bugs in the pre-existing code

- `translate.py:140` — `generate_phrases_recurse` advances `pos + len(word)` but
  `positions` is indexed by Morse symbol. It must advance by `len(code)`. Every
  phrase this printed was misaligned, including `data/7min.all_words.txt`.
- `translate.py:50` — `for letter, code in D:` cannot unpack a `str` key;
  `recurse()` would raise (it is commented out at line 56).
- `audio_analysis.ipynb` cell 19 — `y_slice()` called with no arguments.
- `audio_analysis.ipynb` cell 15/16 — 60 s chunking requests a 3 s overlap but
  the boundary-merge logic is commented out, so the first and last feature of
  every chunk is dropped instead.

### Methodological note on word searching

`data/7min.all_words.txt` (230 KB of dictionary words "found" in the string) is
not evidence of anything. With run-together Morse the number of valid parses
grows exponentially with length, so any dictionary finds thousands of words in
*any* string. Establishing significance requires running the identical search
over shuffled strings with matched symbol frequency and showing the real string
scores outside that null distribution. That was never done, and the search that
produced the file was misaligned anyway (see the `translate.py` bug above).
