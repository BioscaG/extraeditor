/**
 * Timing tokens (§13.1).
 *
 * Durations are defined in **beats**, with a floor in frames. A transition that
 * lasts "a quarter beat" stays musically right at any tempo, but at 160 BPM a
 * quarter beat is 2.8 frames at 30fps — below the ~4 frames the eye needs to read
 * a motion. So every token carries both, and `resolve()` takes the larger.
 *
 * This is why the library never stores raw frame counts: the same component has to
 * work at 24, 25, 30 and 60 fps and at any tempo.
 */

export type BeatDuration = {
  /** Length in beats, resolved against the track's tempo at render time. */
  beats: number;
  /** Hard floor in frames, so fast tempos cannot make the motion unreadable. */
  minFrames: number;
  /** Ceiling in frames, so slow tempos cannot make it feel sluggish. */
  maxFrames?: number;
};

export const timing = {
  /** A hit: the fastest readable motion. Impacts, flash frames. */
  punch: { beats: 0.25, minFrames: 4, maxFrames: 10 } as BeatDuration,
  /** A directional move: whip pans, pushes. Long enough to read the direction. */
  sweep: { beats: 0.5, minFrames: 6, maxFrames: 16 } as BeatDuration,
  /** A settle: zoom punches, beat pulses returning to rest. */
  settle: { beats: 0.75, minFrames: 8, maxFrames: 24 } as BeatDuration,
  /** A breath: dips to white/black, section changes. */
  breath: { beats: 1, minFrames: 10, maxFrames: 32 } as BeatDuration,
  /** A hold: titles and end cards need time to be read. */
  hold: { beats: 4, minFrames: 45, maxFrames: 240 } as BeatDuration,

  /** Word-by-word caption cadence: how long each word holds before the next. */
  wordStep: { beats: 0.25, minFrames: 5, maxFrames: 14 } as BeatDuration,
  /** Text entrance: fast, so the word is legible almost immediately. */
  textIn: { beats: 0.125, minFrames: 3, maxFrames: 8 } as BeatDuration,
} as const;

export type TimingToken = keyof typeof timing;

/** Frames per beat at a given tempo and frame rate. */
export const framesPerBeat = (bpm: number, fps: number): number => (60 / bpm) * fps;

/** Resolve a beat duration to whole frames, honouring its floor and ceiling. */
export const resolve = (duration: BeatDuration, bpm: number, fps: number): number => {
  const raw = Math.round(duration.beats * framesPerBeat(bpm, fps));
  const floored = Math.max(duration.minFrames, raw);
  return duration.maxFrames ? Math.min(duration.maxFrames, floored) : floored;
};

export const resolveToken = (token: TimingToken, bpm: number, fps: number): number =>
  resolve(timing[token], bpm, fps);
