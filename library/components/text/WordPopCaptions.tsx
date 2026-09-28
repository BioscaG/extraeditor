/**
 * text.captions_word_pop — word-by-word captions driven by ASR word timings.
 *
 * Words appear on their own timestamps, not on a fixed cadence: that is what makes
 * captions feel locked to the voice rather than stepped through. The active word is
 * emphasised so the eye tracks the speech.
 *
 * Two things make captions look professional rather than templated:
 *  - the **scrim** is computed from the luminance behind the text and is omitted
 *    entirely when the footage already has enough contrast;
 *  - the block sits inside the platform safe area, so it is never hidden under the
 *    feed UI (§28).
 */

import { fitText } from "@remotion/layout-utils";
import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { palettes, scrimOpacity, textShadow } from "../../tokens/color";
import { settle } from "../../tokens/easing";
import { aspectOf, getInsets } from "../../tokens/safeAreas";
import { timing } from "../../tokens/timing";
import { resolveTextStyle, textStyles } from "../../tokens/type";
import type { ComponentContext, ComponentMeta } from "../contract";

export const wordSchema = z.object({
  text: z.string(),
  /** Seconds relative to the component's start. */
  start: z.number(),
  end: z.number(),
});

export const params = z.object({
  words: z.array(wordSchema).default([]),
  palette: z.enum(["neutral", "festival", "warm"]).default("neutral"),
  /** Vertical placement inside the safe area, 0 = top, 1 = bottom. */
  anchor: z.number().min(0).max(1).default(0.72),
  /** Words visible at once. Keep low: a wall of text is not a caption. */
  windowSize: z.number().int().min(1).max(8).default(4),
  emphasis: z.enum(["scale", "color", "both"]).default("both"),
  uppercase: z.boolean().default(false),
});

export type WordPopParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "text.captions_word_pop",
  version: "1.0.0",
  kind: "text",
  status: "stable",
  duration: { minFrames: 10, maxFrames: 900, default: timing.hold },
  energy: ["low", "medium", "high"],
  tags: ["captions", "speech", "accessibility"],
  beatAnchor: "start",
  sfx: { default: "tick.soft", anchor: "start" },
  params,
  presets: {
    tight: { windowSize: 3, emphasis: "both", uppercase: true },
    readable: { windowSize: 5, emphasis: "color" },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Speech that carries meaning. Not for every line of audible talk — captions on " +
    "filler make the edit feel busy and bury the lines that matter.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<WordPopParams>;
};

export const WordPopCaptions: React.FC<Props> = ({ ctx, params: raw }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  const t = frame / fps;

  if (p.words.length === 0) return null;

  const palette = palettes[p.palette];
  const baseStyle = resolveTextStyle(
    { ...textStyles.caption, uppercase: p.uppercase },
    height,
  );
  const insetsForFit = getInsets(aspectOf(width, height));
  const safeWidth = width * (1 - insetsForFit.left - insetsForFit.right);

  // The longest single word sets the cap: a word wider than the safe area would be
  // clipped by the platform UI, which is the exact failure §28 warns about.
  const longest = p.words.reduce((a, w) => (w.text.length > a.length ? w.text : a), "");
  const fitted = fitText({
    text: p.uppercase ? longest.toUpperCase() : longest,
    withinWidth: safeWidth * 0.8,
    fontFamily: String(baseStyle.fontFamily),
    fontWeight: String(baseStyle.fontWeight),
    letterSpacing: `${Number(baseStyle.letterSpacing)}px`,
  });
  const fontSize = Math.min(Number(baseStyle.fontSize), fitted.fontSize);
  const style: React.CSSProperties = { ...baseStyle, fontSize };

  // The window slides so the active word stays visible without the block jumping.
  const activeIndex = p.words.findIndex((w) => t >= w.start && t < w.end);
  const lastStarted = p.words.reduce(
    (acc, w, i) => (t >= w.start ? i : acc),
    0,
  );
  const cursor = activeIndex >= 0 ? activeIndex : lastStarted;
  const windowStart = Math.max(0, Math.min(
    cursor - Math.floor(p.windowSize / 2),
    p.words.length - p.windowSize,
  ));
  const visible = p.words.slice(windowStart, windowStart + p.windowSize);

  const insets = getInsets(aspectOf(width, height));
  const safeTop = height * insets.top;
  const safeHeight = height * (1 - insets.top - insets.bottom);
  const top = safeTop + safeHeight * p.anchor;

  const scrim = scrimOpacity(ctx.backgroundLuminance ?? 0.5, palette.text);
  const stepFrames = Math.max(timing.textIn.minFrames, Math.round(fps * 0.12));

  return (
    <div
      style={{
        position: "absolute",
        left: width * insets.left,
        width: width * (1 - insets.left - insets.right),
        top,
        transform: "translateY(-50%)",
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "center",
        gap: fontSize * 0.3,
      }}
    >
      {visible.map((word, i) => {
        const index = windowStart + i;
        const startFrame = word.start * fps;
        const age = frame - startFrame;
        if (age < 0) return null;

        const entrance = Math.min(1, age / stepFrames);
        const isActive = index === activeIndex;
        // Pop in with a slight overshoot; active words hold a little larger.
        const scale = settle(entrance, 0.6) * (isActive && p.emphasis !== "color" ? 1.08 : 1);
        const opacity = interpolate(entrance, [0, 1], [0, 1], {
          extrapolateRight: "clamp",
        });
        const color =
          isActive && p.emphasis !== "scale" ? palette.accent : palette.text;

        return (
          <span
            key={`${index}-${word.text}`}
            style={{
              ...style,
              color,
              opacity,
              transform: `scale(${scale.toFixed(4)})`,
              textShadow: textShadow(fontSize, palette.scrim),
              // The scrim is a per-word pill, so it follows the text rather than
              // sitting as a rectangle over the footage.
              backgroundColor: scrim > 0 ? `${palette.scrim}${toHexAlpha(scrim)}` : undefined,
              padding: scrim > 0 ? `${fontSize * 0.08}px ${fontSize * 0.2}px` : undefined,
              borderRadius: scrim > 0 ? fontSize * 0.18 : undefined,
              whiteSpace: "pre",
              willChange: "transform, opacity",
            }}
          >
            {word.text}
          </span>
        );
      })}
    </div>
  );
};

const toHexAlpha = (alpha: number): string =>
  Math.round(Math.min(1, Math.max(0, alpha)) * 255)
    .toString(16)
    .padStart(2, "0");

export default WordPopCaptions;
