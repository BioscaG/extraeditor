/**
 * Component registry.
 *
 * The single place where a versioned plan reference (`transition.whip_pan@1.0.0`)
 * resolves to a component. Plan references are versioned so an old plan renders
 * identically after the library moves on (§17.2); the registry therefore keys on
 * `id@version` and also exposes a latest-version lookup for the director.
 */

import type React from "react";
import type { ComponentMeta } from "./contract";
import BeatPulse, { meta as beatPulseMeta } from "./shot_fx/BeatPulse";
import KineticTitle, { meta as kineticTitleMeta } from "./text/KineticTitle";
import WordPopCaptions, { meta as wordPopMeta } from "./text/WordPopCaptions";
import HardCut, { meta as hardCutMeta } from "./transitions/HardCut";
import WhipPan, { meta as whipPanMeta } from "./transitions/WhipPan";
import ZoomPunch, { meta as zoomPunchMeta } from "./transitions/ZoomPunch";

export type RegistryEntry = {
  meta: ComponentMeta;
  // Props vary by kind (transitions take from/to, fx take children), so the
  // registry stores them loosely and the renderer narrows by `meta.kind`.
  Component: React.FC<any>;
};

const entries: RegistryEntry[] = [
  { meta: hardCutMeta, Component: HardCut },
  { meta: whipPanMeta, Component: WhipPan },
  { meta: zoomPunchMeta, Component: ZoomPunch },
  { meta: wordPopMeta, Component: WordPopCaptions },
  { meta: kineticTitleMeta, Component: KineticTitle },
  { meta: beatPulseMeta, Component: BeatPulse },
];

export const registry: Record<string, RegistryEntry> = Object.fromEntries(
  entries.map((e) => [`${e.meta.id}@${e.meta.version}`, e]),
);

/** All versions of a component id, newest first. */
const byId = (id: string): RegistryEntry[] =>
  entries
    .filter((e) => e.meta.id === id)
    .sort((a, b) => compareVersions(b.meta.version, a.meta.version));

/**
 * Resolve a plan reference. Accepts `id@version`, `id@major.minor` or a bare `id`.
 *
 * A partial version matches the newest release with that prefix, so a plan pinned to
 * `@1.0` picks up `1.0.3` bug fixes but never `1.1`.
 */
export const resolve = (ref: string): RegistryEntry | undefined => {
  if (registry[ref]) return registry[ref];
  const [id, version] = ref.split("@");
  const candidates = byId(id);
  if (candidates.length === 0) return undefined;
  if (!version) return candidates[0];
  return candidates.find((e) => e.meta.version === version || e.meta.version.startsWith(`${version}.`));
};

const compareVersions = (a: string, b: string): number => {
  const pa = a.split(".").map(Number);
  const pb = b.split(".").map(Number);
  for (let i = 0; i < 3; i++) {
    if ((pa[i] ?? 0) !== (pb[i] ?? 0)) return (pa[i] ?? 0) - (pb[i] ?? 0);
  }
  return 0;
};

export const allMeta = (): ComponentMeta[] => entries.map((e) => e.meta);

export const stableMeta = (): ComponentMeta[] =>
  entries.filter((e) => e.meta.status === "stable").map((e) => e.meta);

export * from "./contract";
export { BeatPulse, HardCut, KineticTitle, WhipPan, WordPopCaptions, ZoomPunch };
