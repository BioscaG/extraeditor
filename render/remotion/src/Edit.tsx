/**
 * The Edit composition: turns an EditPlan into frames (§19.3).
 *
 * Structure:
 *  - each shot is a `Sequence` at its `timeline_in`, playing its conformed
 *    intermediate through `OffthreadVideo` (frame-accurate, unlike `<Video>`);
 *  - a shot's `transition_in` overlaps the *previous* shot, so transitions render in
 *    their own layer above both;
 *  - captions and overlays sit on top, inside the safe area.
 *
 * Video is rendered **muted**: the ffmpeg mix is muxed in afterwards (§19.3), because
 * ducking envelopes and two-pass loudness belong in the audio graph, not here.
 */

import React from "react";
import { AbsoluteFill, OffthreadVideo, Sequence, useVideoConfig } from "remotion";
import { registry, resolve } from "../../../library/components";
import type { ComponentContext } from "../../../library/components/contract";
import { resolve as resolveTiming } from "../../../library/tokens/timing";
import type { ComponentRef, Edit, ResolvedShot, Shot } from "./schema";

type ShotPair = { shot: Shot; resolved: ResolvedShot };

export const EditComposition: React.FC<Edit> = (edit) => {
  const { width, height, fps } = useVideoConfig();
  const byId = new Map(edit.resolved.map((r) => [r.shot_id, r]));
  const pairs: ShotPair[] = edit.shots
    .slice()
    .sort((a, b) => a.timeline_in - b.timeline_in)
    .flatMap((shot) => {
      const resolved = byId.get(shot.id);
      // A shot with no intermediate is skipped rather than rendered black: the
      // conform step reports the failure, and one bad clip must not void the edit.
      return resolved ? [{ shot, resolved }] : [];
    });

  const baseCtx = { width, height, fps, bpm: edit.bpm };

  return (
    <AbsoluteFill style={{ backgroundColor: "black" }}>
      {pairs.map(({ shot, resolved }, index) => {
        const transition = shot.transition_in ?? null;
        const transitionFrames = transition
          ? transitionLength(transition, edit.bpm, fps)
          : 0;
        const previous = index > 0 ? pairs[index - 1] : null;

        return (
          <React.Fragment key={shot.id}>
            <Sequence
              from={shot.timeline_in}
              durationInFrames={resolved.duration_in_frames}
              layout="none"
              name={`${shot.id} ${shot.intent}`.trim()}
            >
              <ShotLayer shot={shot} resolved={resolved} ctx={baseCtx} edit={edit} />
            </Sequence>

            {transition && previous && transitionFrames > 0 ? (
              <Sequence
                // Centred on the cut: half the transition eats into the outgoing shot.
                from={Math.max(0, shot.timeline_in - Math.floor(transitionFrames / 2))}
                durationInFrames={transitionFrames}
                layout="none"
                name={`${shot.id} ${transition.id}`}
              >
                <TransitionLayer
                  ref={transition}
                  from={previous}
                  to={{ shot, resolved }}
                  ctx={{ ...baseCtx, frame: 0, durationInFrames: transitionFrames }}
                  edit={edit}
                />
              </Sequence>
            ) : null}
          </React.Fragment>
        );
      })}

      {edit.overlays.map((overlay) => {
        const entry = resolve(overlay.component);
        if (!entry) return null;
        const duration = Math.max(1, overlay.to_frame - overlay.from_frame);
        return (
          <Sequence
            key={overlay.id}
            from={overlay.from_frame}
            durationInFrames={duration}
            layout="none"
            name={`${overlay.id} ${overlay.component}`}
          >
            <entry.Component
              ctx={{ ...baseCtx, frame: 0, durationInFrames: duration }}
              params={{ palette: edit.palette, ...overlay.props }}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};

/** One shot: the intermediate, its shot effects, and its captions. */
const ShotLayer: React.FC<{
  shot: Shot;
  resolved: ResolvedShot;
  ctx: Omit<ComponentContext, "frame" | "durationInFrames">;
  edit: Edit;
}> = ({ shot, resolved, ctx, edit }) => {
  const fullCtx: ComponentContext = {
    ...ctx,
    frame: 0,
    durationInFrames: resolved.duration_in_frames,
    backgroundLuminance: resolved.background_luminance,
  };

  // The intermediate begins `head_handle_s` before the shot's in-point (§19.1), so
  // playback must start there or the whole edit sits early.
  let content: React.ReactNode = (
    <OffthreadVideo
      src={resolved.src}
      startFrom={Math.round(resolved.head_handle_s * ctx.fps)}
      playbackRate={constantRate(shot)}
      muted
      style={{ width: "100%", height: "100%", objectFit: "cover" }}
    />
  );

  // Shot effects wrap the picture, innermost first, so `fx` order is composition order.
  for (const fx of shot.fx) {
    const entry = resolve(fx.id);
    if (!entry) continue;
    const props = { ...fx.props };
    if (fx.on === "downbeats" || fx.on === "beats") {
      props.beats = resolved.beats;
    }
    content = (
      <entry.Component ctx={fullCtx} params={props}>
        {content}
      </entry.Component>
    );
  }

  return (
    <AbsoluteFill>
      {content}
      {shot.captions ? (
        <CaptionLayer captions={shot.captions} resolved={resolved} ctx={fullCtx} edit={edit} />
      ) : null}
    </AbsoluteFill>
  );
};

const CaptionLayer: React.FC<{
  captions: NonNullable<Shot["captions"]>;
  resolved: ResolvedShot;
  ctx: ComponentContext;
  edit: Edit;
}> = ({ captions, resolved, ctx, edit }) => {
  const entry = resolve(captions.id);
  if (!entry || resolved.words.length === 0) return null;
  return (
    <entry.Component ctx={ctx} params={{ words: resolved.words, palette: edit.palette }} />
  );
};

const TransitionLayer: React.FC<{
  ref: ComponentRef;
  from: ShotPair;
  to: ShotPair;
  ctx: ComponentContext;
  edit: Edit;
}> = ({ ref, from, to, ctx }) => {
  const entry = resolve(ref.id);
  // An unknown transition degrades to a hard cut rather than failing the render.
  if (!entry || entry.meta.id === "transition.hard_cut") return null;

  const still = (pair: ShotPair, offsetFrames: number) => (
    <OffthreadVideo
      src={pair.resolved.src}
      startFrom={Math.max(0, Math.round(pair.resolved.head_handle_s * ctx.fps) + offsetFrames)}
      muted
      style={{ width: "100%", height: "100%", objectFit: "cover" }}
    />
  );

  return (
    <entry.Component
      ctx={ctx}
      params={{ ...ref.props, ...(ref.preset ? entry.meta.presets?.[ref.preset] : {}) }}
      // The outgoing shot is sampled near its tail, the incoming one at its head.
      from={still(from, Math.max(0, from.resolved.duration_in_frames - ctx.durationInFrames))}
      to={still(to, 0)}
    />
  );
};

/** Resolve a transition's declared duration to frames, clamped to its own limits. */
const transitionLength = (ref: ComponentRef, bpm: number, fps: number): number => {
  const entry = resolve(ref.id);
  if (!entry) return 0;
  const { minFrames, maxFrames, default: fallback } = entry.meta.duration;
  if (ref.duration?.frames != null) {
    return Math.min(maxFrames, Math.max(minFrames, ref.duration.frames));
  }
  if (ref.duration?.beats != null) {
    return resolveTiming({ ...fallback, beats: ref.duration.beats }, bpm, fps);
  }
  return resolveTiming(fallback, bpm, fps);
};

/**
 * A single constant speed maps to `playbackRate`. Piecewise ramps are resolved into
 * separate shots by the plan before reaching the renderer, so anything else is unit speed.
 */
const constantRate = (shot: Shot): number =>
  shot.speed.length === 1 ? shot.speed[0].rate : 1;

export default EditComposition;
export { registry };
