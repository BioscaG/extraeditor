/**
 * text.stamp — a time-and-place stamp, as in "18:40 — MAIN STAGE".
 *
 * Small, monospaced, wide-tracked and in a corner. It works because it is *information*
 * rather than decoration: it tells the viewer where they are in the day, which is exactly
 * what an aftermovie of one long event needs and what a title cannot do.
 *
 * The rule it follows that most stamps break: it does not animate after it has arrived.
 * A stamp that keeps moving competes with the footage; one that appears and sits still
 * reads as a caption on a photograph.
 */

import { fitText } from "@remotion/layout-utils";
import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { palettes, scrimOpacity } from "../../tokens/color";
import { settle } from "../../tokens/easing";
import { aspectOf, getInsets } from "../../tokens/safeAreas";
import { timing } from "../../tokens/timing";
import { resolveTextStyle, textStyles } from "../../tokens/type";
import type { ComponentContext, ComponentMeta } from "../contract";
import { progressOf } from "../contract";

export const params = z.object({
  /** Left-hand part, conventionally a time. */
  primary: z.string().default(""),
  /** Right-hand part, conventionally a place. */
  secondary: z.string().default(""),
  palette: z.enum(["neutral", "festival", "warm"]).default("neutral"),
  corner: z.enum(["top-left", "top-right", "bottom-left", "bottom-right"])
    .default("top-left"),
  /** Draw a rule between the two parts. */
  divider: z.boolean().default(true),
});

export type StampParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "text.stamp",
  version: "1.0.0",
  kind: "text",
  status: "stable",
  duration: { minFrames: 20, maxFrames: 240, default: timing.hold },
  energy: ["low", "medium"],
  tags: ["stamp", "time", "place", "information"],
  beatAnchor: "start",
  sfx: { default: "tick.soft", anchor: "start" },
  params,
  presets: {
    time: { corner: "top-left", divider: true },
    place: { corner: "bottom-left", divider: false },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Orienting the viewer in a long event — the first shot of a new location or a new " +
    "part of the day. It carries information, so use it where the viewer would otherwise " +
    "be lost, not as decoration on a shot that needs none.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<StampParams>;
};

export const Stamp: React.FC<Props> = ({ ctx, params: raw }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  if (!p.primary && !p.secondary) return null;

  const palette = palettes[p.palette];
  const baseStyle = resolveTextStyle(textStyles.stamp, height);
  const insets = getInsets(aspectOf(width, height));
  const safeWidth = width * (1 - insets.left - insets.right);

  // Wide tracking is what makes a stamp read as a stamp, and it is also what makes it
  // overflow: "18:40 — MAIN STAGE" at the caption size ran off the right edge of a 9:16
  // frame. Both parts are measured as one string and the whole stamp is scaled to fit.
  const measured = `${p.primary}${p.secondary}`;
  const fitted = fitText({
    text: measured,
    withinWidth: safeWidth,
    fontFamily: String(baseStyle.fontFamily),
    fontWeight: String(baseStyle.fontWeight),
    letterSpacing: `${Number(baseStyle.letterSpacing)}px`,
  });

  // The pill's padding, the two gaps and the divider all scale with the font size, so
  // fitting the *text* to the safe width still overflows by their total. Text width is
  // linear in font size, so with `k = safeWidth / fittedSize` the size that leaves room
  // for `chrome` ems of extra width solves exactly.
  const chromeEms =
    1.4 /* pill padding */ +
    (p.primary && p.secondary ? 1.4 /* two gaps */ + (p.divider ? 1.4 : 0) : 0);
  const budgeted =
    (fitted.fontSize * safeWidth) / (safeWidth + chromeEms * fitted.fontSize);
  const scale = Math.min(1, budgeted / Number(baseStyle.fontSize));
  const fontSize = Number(baseStyle.fontSize) * scale;
  const style: React.CSSProperties = {
    ...baseStyle,
    fontSize,
    letterSpacing: Number(baseStyle.letterSpacing) * scale,
  };

  const inFrames = Math.max(timing.textIn.minFrames, Math.round(fps * 0.25));
  const entrance = Math.min(1, Math.max(0, frame / inFrames));
  const progress = progressOf({ ...ctx, frame });
  const exit = interpolate(progress, [0.88, 1], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // It arrives, then holds. Nothing moves after `entrance` reaches 1.
  const slide = (1 - settle(entrance, 0.85)) * fontSize * 0.6;
  const top = p.corner.startsWith("top");
  const left = p.corner.endsWith("left");
  const scrim = scrimOpacity(ctx.backgroundLuminance ?? 0.5, palette.text);

  return (
    <div
      style={{
        position: "absolute",
        [top ? "top" : "bottom"]: height * (top ? insets.top : insets.bottom),
        [left ? "left" : "right"]: width * (left ? insets.left : insets.right),
        display: "flex",
        alignItems: "center",
        // Cap the width so a long place name wraps or clips inside the safe area rather
        // than running off the frame.
        maxWidth: safeWidth,
        gap: fontSize * 0.7,
        transform: `translateX(${(left ? -slide : slide).toFixed(2)}px)`,
        opacity: Math.min(entrance, exit),
        backgroundColor: scrim > 0 ? `${palette.scrim}${toHexAlpha(scrim * 0.75)}` : undefined,
        padding: scrim > 0 ? `${fontSize * 0.4}px ${fontSize * 0.7}px` : undefined,
        borderRadius: scrim > 0 ? fontSize * 0.3 : undefined,
        willChange: "transform, opacity",
      }}
    >
      {p.primary ? (
        <span style={{ ...style, color: palette.accent, whiteSpace: "pre" }}>
          {p.primary}
        </span>
      ) : null}
      {p.divider && p.primary && p.secondary ? (
        <span
          style={{
            width: fontSize * 1.4,
            height: Math.max(1, fontSize * 0.08),
            backgroundColor: palette.text,
            opacity: 0.6,
            // Without this the rule vanishes: it is a fixed-width flex item inside a
            // max-width container, so the default `flex-shrink: 1` collapses it to nothing.
            flexShrink: 0,
          }}
        />
      ) : null}
      {p.secondary ? (
        <span style={{ ...style, color: palette.text, whiteSpace: "pre" }}>
          {p.secondary}
        </span>
      ) : null}
    </div>
  );
};

const toHexAlpha = (alpha: number): string =>
  Math.round(Math.min(1, Math.max(0, alpha)) * 255)
    .toString(16)
    .padStart(2, "0");

export default Stamp;
