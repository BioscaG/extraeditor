/**
 * Remotion root: the compositions the CLI can render.
 *
 *  - `Edit`      the real edit, driven entirely by input props from the Python side.
 *  - `Gallery`   one component × preset × aspect ratio, for library review (§13.4).
 *
 * Dimensions and duration come from input props via `calculateMetadata`, because the
 * project's format is a plan decision, not a build-time constant.
 */

import React from "react";
import { Composition } from "remotion";
import { EditComposition } from "./Edit";
import { GalleryComposition, gallerySchema } from "./Gallery";
import { editSchema } from "./schema";

const FALLBACK = {
  version: 0,
  project: "empty",
  format: { width: 1080, height: 1920, fps: 30, dynamic_range: "sdr" },
  concept: { title: "empty", sections: [] },
  shots: [],
  overlays: [],
  bpm: 120,
  resolved: [],
  palette: "neutral" as const,
};

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="Edit"
      component={EditComposition}
      schema={editSchema}
      defaultProps={FALLBACK}
      // A composition must declare something; the real values arrive per render.
      durationInFrames={30}
      fps={30}
      width={1080}
      height={1920}
      calculateMetadata={({ props }) => {
        const end = Math.max(
          1,
          ...props.shots.map((s) => {
            const resolved = props.resolved.find((r) => r.shot_id === s.id);
            return s.timeline_in + (resolved?.duration_in_frames ?? 0);
          }),
          ...props.overlays.map((o) => o.to_frame),
        );
        return {
          durationInFrames: end,
          fps: props.format.fps,
          width: props.format.width,
          height: props.format.height,
        };
      }}
    />

    <Composition
      id="Gallery"
      component={GalleryComposition}
      schema={gallerySchema}
      defaultProps={{
        componentId: "transition.whip_pan@1.0.0",
        preset: null,
        params: {},
        bpm: 120,
        durationInFrames: 30,
        label: "",
      }}
      durationInFrames={30}
      fps={30}
      width={1080}
      height={1920}
      calculateMetadata={({ props }) => ({
        durationInFrames: Math.max(1, props.durationInFrames),
      })}
    />
  </>
);

export default RemotionRoot;
