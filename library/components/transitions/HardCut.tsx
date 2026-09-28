/**
 * transition.hard_cut — the default, and by far the most frequent.
 *
 * It exists as a component so that "no transition" is an explicit, countable
 * decision in the plan rather than an absence. The rhythm report counts designed
 * transitions against a style limit (§18.3); having the hard cut in the library is
 * what lets the director say "cut" rather than reach for an effect.
 *
 * Renders nothing: the cut is the absence of overlap.
 */

import React from "react";
import { z } from "zod";
import type { ComponentContext, ComponentMeta } from "../contract";

export const params = z.object({});

export const meta: ComponentMeta<typeof params> = {
  id: "transition.hard_cut",
  version: "1.0.0",
  kind: "transition",
  status: "stable",
  duration: { minFrames: 0, maxFrames: 0, default: { beats: 0, minFrames: 0 } },
  energy: ["low", "medium", "high"],
  tags: ["default", "invisible"],
  beatAnchor: "cut_point",
  params,
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "The default. Use it unless a designed transition earns its place — transition " +
    "overuse is the most recognisable amateur tell.",
};

export const HardCut: React.FC<{ ctx: ComponentContext }> = () => null;

export default HardCut;
