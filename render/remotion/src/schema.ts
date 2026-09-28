/**
 * TypeScript mirror of the EditPlan (§17).
 *
 * The pydantic model in `montaje.models.editplan` is the source of truth; this is
 * the render side's view of it, plus the resolved intermediates the Python side
 * supplies. A schema-sync test fails CI if the two drift.
 *
 * Only what the renderer needs appears here — colour parameters, for instance, are
 * already baked into the intermediates by the conform step (§19.1), so the renderer
 * never sees them.
 */

import { z } from "zod";

export const durationSchema = z.object({
  beats: z.number().nullable().optional(),
  frames: z.number().int().nullable().optional(),
});

export const componentRefSchema = z.object({
  id: z.string(),
  preset: z.string().nullable().optional(),
  duration: durationSchema.nullable().optional(),
  props: z.record(z.string(), z.unknown()).default({}),
  on: z.string().nullable().optional(),
});

export const speedRampSchema = z.object({
  from: z.number(),
  to: z.number(),
  rate: z.number().positive(),
});

export const shotSchema = z.object({
  id: z.string(),
  asset: z.string(),
  src_in: z.number(),
  src_out: z.number(),
  timeline_in: z.number().int(),
  speed: z.array(speedRampSchema).default([]),
  reframe: z
    .object({
      mode: z.string().default("center"),
      ease: z.string().default("glide"),
      rect: z.array(z.number()).nullable().optional(),
    })
    .default({ mode: "center", ease: "glide" }),
  transition_in: componentRefSchema.nullable().optional(),
  fx: z.array(componentRefSchema).default([]),
  captions: z
    .object({ id: z.string(), words: z.string().default("auto") })
    .nullable()
    .optional(),
  section: z.string().nullable().optional(),
  intent: z.string().default(""),
});

export const overlaySchema = z.object({
  id: z.string(),
  component: z.string(),
  from_frame: z.number().int(),
  to_frame: z.number().int(),
  props: z.record(z.string(), z.unknown()).default({}),
});

export const sectionSchema = z.object({
  id: z.string(),
  from_frame: z.number().int(),
  to_frame: z.number().int(),
  music_section: z.string().nullable().optional(),
  purpose: z.string().nullable().optional(),
});

/** What the Python side resolves for each shot before invoking Remotion. */
export const resolvedShotSchema = z.object({
  shot_id: z.string(),
  /** Intermediate file, already colour-graded and conformed (§19.1). */
  src: z.string(),
  /**
   * Seconds of handle material at the head of the intermediate.
   *
   * The intermediate starts before the shot's in-point, so the renderer must seek
   * past the handle. Without this the whole edit would sit half a second early.
   */
  head_handle_s: z.number(),
  /** Length on the timeline in frames, after speed ramps. */
  duration_in_frames: z.number().int(),
  /** Mean luminance behind any text on this shot, 0–1, for scrim decisions. */
  background_luminance: z.number().min(0).max(1).default(0.5),
  /** Beat times in seconds relative to the shot start, for beat-locked fx. */
  beats: z.array(z.number()).default([]),
  /** ASR words within the shot, times relative to the shot start. */
  words: z
    .array(z.object({ text: z.string(), start: z.number(), end: z.number() }))
    .default([]),
});

export const editSchema = z.object({
  version: z.number().int(),
  project: z.string(),
  format: z.object({
    width: z.number().int(),
    height: z.number().int(),
    fps: z.number(),
    dynamic_range: z.string().default("sdr"),
  }),
  style: z.string().nullable().optional(),
  concept: z.object({
    title: z.string(),
    logline: z.string().nullable().optional(),
    signature: z.string().nullable().optional(),
    sections: z.array(sectionSchema).default([]),
  }),
  shots: z.array(shotSchema).default([]),
  overlays: z.array(overlaySchema).default([]),
  /** Tempo of the final music, needed to resolve beat-based durations. */
  bpm: z.number().default(120),
  resolved: z.array(resolvedShotSchema).default([]),
  /** Palette name applied to every text component unless overridden per-props. */
  palette: z.enum(["neutral", "festival", "warm"]).default("neutral"),
});

export type Edit = z.infer<typeof editSchema>;
export type Shot = z.infer<typeof shotSchema>;
export type ResolvedShot = z.infer<typeof resolvedShotSchema>;
export type Overlay = z.infer<typeof overlaySchema>;
export type ComponentRef = z.infer<typeof componentRefSchema>;
