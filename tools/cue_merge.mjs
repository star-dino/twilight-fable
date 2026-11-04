/**
 * Pure cue merge for `tools/render.mjs`: authored DOM cues + raw runtime cues -> the final cue list.
 *
 * Runtime cues (`registerRuntimeCue`, engine/motion.js) are recorded raw, one event per rendered frame that satisfies
 * the trigger; nothing is decided at emit time. The final list is therefore a pure function of the set of frame
 * times, independent of how frames were spread over worker pages or in what order they were drawn.
 *
 * Rules, applied in time order (ties broken by kind/label/scene, never by arrival order):
 *   1. authored cues are seeded first and always win over runtime cues within the refractory window;
 *   2. a footstep (paw_step / foot_step) within STEP_HOP_GAP of a hop (critter_hop / boing_hop / cartoon_boing) in the
 *      same scene is part of that hop's take-off / landing and is dropped (the hop cue already carries the foley);
 *   3. refractory window: same locomotion kind 0.42 s, DOM kinds 0.08 s, other kinds same (kind,label) 0.42 s.
 */
const STEP_KINDS = new Set(['paw_step', 'foot_step']);
const HOP_KINDS = new Set(['critter_hop', 'boing_hop', 'cartoon_boing']);
const LOCO_KINDS = new Set(['paw_step', 'foot_step', 'critter_hop']);
const DOM_KINDS = new Set(['scene_cut', 'card_enter', 'stroke_draw', 'count_tick', 'layer_float']);
export const STEP_HOP_GAP = 0.42;

const order = (a, b) =>
  a.t - b.t ||
  String(a.kind).localeCompare(String(b.kind)) ||
  String(a.label || '').localeCompare(String(b.label || '')) ||
  String(a.scene || '').localeCompare(String(b.scene || ''));

export function mergeCues(authoredIn, runtimeRaw) {
  const authored = [...authoredIn].sort(order);
  const runtime = [...runtimeRaw].sort(order);
  const hops = [...authored, ...runtime].filter((c) => HOP_KINDS.has(c.kind));
  const out = [];
  for (const c of [...authored, ...runtime.filter((r) => !(STEP_KINDS.has(r.kind) && hops.some((h) => h.scene === r.scene && Math.abs(h.t - r.t) < STEP_HOP_GAP)))]) {
    const isLoco = LOCO_KINDS.has(c.kind);
    const minGap = isLoco ? 0.42 : DOM_KINDS.has(c.kind) ? 0.08 : 0.42;
    const id = `${c.kind}:${c.label || c.kind}`;
    const dup = out.some((p) => (isLoco ? p.kind === c.kind : `${p.kind}:${p.label || p.kind}` === id) && Math.abs(p.t - c.t) < minGap);
    if (!dup) out.push(c);
  }
  return out.sort(order);
}
