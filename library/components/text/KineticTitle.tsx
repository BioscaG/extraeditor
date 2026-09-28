/**
 * text.kinetic_title — a display title that animates in per line.
 *
 * Lines enter on a stagger, each masked by its own bounding box so the type appears
 * to be revealed rather than to fly in. Masked reveals read as designed; sliding
 * whole words reads as a template.
 *
 * The stagger is in beats, so a title lands with the music at any tempo.
 */

import { fitText } from "@remotion/layout-utils";
import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { palettes, scrimOpacity } from "../../tokens/color";
import { easing, settle } from "../../tokens/easing";
import { aspectOf, getInsets } from "../../tokens/safeAreas";
import { framesPerBeat, timing } from "../../tokens/timing";
import { resolveTextStyle, textStyles } from "../../tokens/type";
import type { ComponentContext, ComponentMeta } from "../contract";
import { progressOf } from "../contract";

export const params = z.object({
  lines: z.array(z.string()).min(1).default(["TITLE"]),
  palette: z.enum(["neutral", "festival", "warm"]).default("neutral"),
  align: z.enum(["left", "center", "right"]).default("center"),
  /** Vertical placement inside the safe area, 0 = top, 1 = bottom. */
  anchor: z.number().min(0).max(1).default(0.5),
  /** Stagger between lines, in beats. */
  staggerBeats: z.number().min(0).max(1).default(0.25),
  /** Which line (0-based) gets the accent colour. -1 for none. */
  accentLine: z.number().int().default(-1),
  exit: z.enum(["fade", "mask", "hold"]).default("fade"),
});

export type KineticTitleParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "text.kinetic_title",
  version: "1.0.0",
  kind: "text",
  status: "stable",
  duration: { minFrames: 20, maxFrames: 300, default: timing.hold },
  energy: ["medium", "high"],
  tags: ["title", "display", "typography"],
  beatAnchor: "start",
  sfx: { default: "whoosh.soft", anchor: "start" },
  params,
  presets: {
    opener: { align: "center", anchor: 0.45, staggerBeats: 0.25 },
    lowerLeft: { align: "left", anchor: 0.78, staggerBeats: 0.125 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "The opening title and section cards. One per edit, at most two: titles compete " +
    "with the footage for attention.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<KineticTitleParams>;
};

export const KineticTitle: React.FC<Props> = ({ ctx, params: raw }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  const progress = progressOf({ ...ctx, frame });

  const palette = palettes[p.palette];
  const baseStyle = resolveTextStyle(textStyles.title, height);
  const insets = getInsets(aspectOf(width, height));
  const safeTop = height * insets.top;
  const safeHeight = height * (1 - insets.top - insets.bottom);
  const safeWidth = width * (1 - insets.left - insets.right);

  // The display size from the type scale is a *maximum*. A long line at that size
  // overflows a 9:16 frame, so every line is measured and the whole block is scaled
  // down by the worst-fitting one — scaling per line would break the type hierarchy.
  const fitScale = Math.min(
    1,
    ...p.lines.map((line) => {
      const measured = fitText({
        text: line.toUpperCase(),
        withinWidth: safeWidth,
        fontFamily: String(baseStyle.fontFamily),
        fontWeight: String(baseStyle.fontWeight),
        letterSpacing: `${Number(baseStyle.letterSpacing)}px`,
      });
      return measured.fontSize / Number(baseStyle.fontSize);
    }),
  );
  const fontSize = Number(baseStyle.fontSize) * fitScale;
  const style: React.CSSProperties = {
    ...baseStyle,
    fontSize,
    letterSpacing: Number(baseStyle.letterSpacing) * fitScale,
  };

  const staggerFrames = p.staggerBeats * framesPerBeat(ctx.bpm, fps);
  const revealFrames = Math.max(timing.sweep.minFrames, Math.round(fps * 0.35));

  // Exit begins in the last 20% so a `hold` title still clears before the next shot.
  const exitProgress = interpolate(progress, [0.8, 1], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: easing.ease,
  });
  const exitOpacity = p.exit === "fade" ? 1 - exitProgress : 1;

  const scrim = scrimOpacity(ctx.backgroundLuminance ?? 0.5, palette.text);

  return (
    <div
      style={{
        position: "absolute",
        left: width * insets.left,
        width: width * (1 - insets.left - insets.right),
        top: safeTop + safeHeight * p.anchor,
        transform: "translateY(-50%)",
        display: "flex",
        flexDirection: "column",
        alignItems:
          p.align === "center" ? "center" : p.align === "left" ? "flex-start" : "flex-end",
        gap: fontSize * 0.08,
        opacity: exitOpacity,
      }}
    >
      {p.lines.map((line, i) => {
        const lineStart = i * staggerFrames;
        const local = Math.min(1, Math.max(0, (frame - lineStart) / revealFrames));
        // Masked reveal: the line slides up inside its own clipping box, so the type
        // is uncovered rather than flown in.
        const shift = (1 - settle(local, 0.8)) * fontSize * 1.1;
        const maskExit = p.exit === "mask" ? exitProgress * fontSize * 1.1 : 0;

        return (
          <div
            key={`${i}-${line}`}
            style={{
              overflow: "hidden",
              // A touch of extra height so descenders are not clipped by the mask.
              paddingBottom: fontSize * 0.12,
              marginBottom: -fontSize * 0.12,
              backgroundColor: scrim > 0 ? `${palette.scrim}${toHexAlpha(scrim * 0.8)}` : undefined,
              padding: scrim > 0 ? `0 ${fontSize * 0.12}px` : undefined,
            }}
          >
            <div
              style={{
                ...style,
                color: i === p.accentLine ? palette.accent : palette.text,
                transform: `translateY(${(shift - maskExit).toFixed(2)}px)`,
                opacity: local > 0 ? 1 : 0,
                whiteSpace: "pre",
                willChange: "transform",
              }}
            >
              {line}
            </div>
          </div>
        );
      })}
    </div>
  );
};

const toHexAlpha = (alpha: number): string =>
  Math.round(Math.min(1, Math.max(0, alpha)) * 255)
    .toString(16)
    .padStart(2, "0");

export default KineticTitle;
