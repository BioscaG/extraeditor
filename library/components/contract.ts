/**
 * The component contract (§13.2).
 *
 * Every library component exports `meta` conforming to `ComponentMeta` and a React
 * component. The rules that make components composable and cacheable:
 *
 *  - animation is driven **only** by `useCurrentFrame()`, never by `Date.now()` or
 *    React state, so a frame renders identically however it is reached;
 *  - randomness is seeded via Remotion's `random()`, never `Math.random()`;
 *  - no network assets; fonts and textures come from `library/` only;
 *  - must render correctly at 24/25/30/60 fps and in every declared aspect ratio,
 *    which is why durations are beats-and-frames tokens rather than frame counts;
 *  - sizes derive from frame dimensions and design tokens, never from pixel literals.
 *
 * The Python mirror of this type is `montaje.models.library.ComponentMeta`; a schema
 * sync test keeps them aligned.
 */

import { z } from "zod";
import type { BeatDuration } from "../tokens/timing";

export type ComponentKind =
  | "transition"
  | "shot_fx"
  | "text"
  | "overlay"
  | "finishing"
  | "layout";

export type ComponentStatus = "draft" | "candidate" | "stable" | "deprecated";

export type Energy = "low" | "medium" | "high";

/** Which instant of the component should land on the beat. */
export type BeatAnchor = "start" | "cut_point" | "peak" | "end";

/** Which shot motion direction the component prefers to follow. */
export type MotionMatch = "horizontal" | "vertical" | "zoom_in" | "zoom_out" | "any";

export type ComponentMeta<P extends z.ZodTypeAny = z.ZodTypeAny> = {
  id: string;
  version: string;
  kind: ComponentKind;
  status: ComponentStatus;
  duration: {
    minFrames: number;
    maxFrames: number;
    default: BeatDuration;
  };
  energy: Energy[];
  tags: string[];
  beatAnchor?: BeatAnchor;
  motionMatch?: MotionMatch;
  sfx?: { default: string; anchor: BeatAnchor };
  params: P;
  /** Named parameter presets. Good presets matter more than infinite parameters. */
  presets?: Record<string, Partial<z.infer<P>>>;
  aspectRatios: Array<"9:16" | "16:9" | "1:1" | "4:5">;
  author: "human" | "agent";
  /** One line the director reads when choosing. Say when *not* to use it too. */
  intent: string;
};

/** Props every component receives in addition to its own `params`. */
export type ComponentContext = {
  /** Frames from the component's own start, so components never see absolute time. */
  frame: number;
  /** The component's total length in frames, already resolved from beats. */
  durationInFrames: number;
  width: number;
  height: number;
  fps: number;
  /** Track tempo, for components that subdivide the beat internally. */
  bpm: number;
  /** Mean luminance behind the component, 0–1, for scrim decisions. */
  backgroundLuminance?: number;
};

/** Progress through the component, 0–1. The basis of every animation. */
export const progressOf = (ctx: ComponentContext): number =>
  ctx.durationInFrames <= 1 ? 1 : Math.min(1, Math.max(0, ctx.frame / (ctx.durationInFrames - 1)));

/**
 * Progress of a transition measured from its midpoint, −1 → +1.
 *
 * Transitions are centred on the cut: the outgoing shot occupies the negative half
 * and the incoming shot the positive half.
 */
export const signedProgressOf = (ctx: ComponentContext): number =>
  progressOf(ctx) * 2 - 1;
