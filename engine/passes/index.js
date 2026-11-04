import grain from './grain.js';

export const PASSES = { grain };

/**
 * Resolve a style's `passes` array into stage input.
 * Supports tuple syntax `[['grain', { amount: 0.008 }]]`
 * as well as object syntax `[{ pass: 'grain' | grainObj, amount: 0.008 }]`.
 */
export function resolvePasses(list = []) {
  return list.map((entry) => {
    if (Array.isArray(entry)) {
      const [nameOrObj, params = {}] = entry;
      const pass = typeof nameOrObj === 'string' ? PASSES[nameOrObj] : nameOrObj;
      if (!pass) throw new Error(`Unknown pass "${nameOrObj}". Available: ${Object.keys(PASSES).join(', ')}`);
      return { pass, params };
    }
    if (entry && typeof entry === 'object' && 'pass' in entry) {
      const { pass: nameOrObj, params: explicitParams, ...rest } = entry;
      const pass = typeof nameOrObj === 'string' ? PASSES[nameOrObj] : nameOrObj;
      if (!pass) throw new Error(`Unknown pass "${nameOrObj}". Available: ${Object.keys(PASSES).join(', ')}`);
      return { pass, params: explicitParams ? { ...rest, ...explicitParams } : rest };
    }
    const pass = typeof entry === 'string' ? PASSES[entry] : entry;
    if (!pass) throw new Error(`Unknown pass "${entry}". Available: ${Object.keys(PASSES).join(', ')}`);
    return { pass, params: {} };
  });
}
