// parity_deno.mjs — kernel-level parity harness on Deno's WebGPU (lavapipe)
// Usage: deno run -A --unstable-webgpu parity_deno.mjs <dir> [mode]
import { Runtime } from "./rt.js";
import { readFileSync } from "node:fs";

const DIR = Deno.args[0] || ".";
const MODE = Deno.args[1] || "kernel";
const TOL = 1e-4, REL = 3e-6;
const tolOf = (w) => TOL + REL * Math.abs(w);

const doc = JSON.parse(readFileSync(`${DIR}/kernels.json`, "utf8"));
const wBuf = readFileSync(`${DIR}/weights.bin`);
const wBytes = wBuf.buffer.slice(wBuf.byteOffset, wBuf.byteOffset + wBuf.byteLength);
const featsJ = JSON.parse(readFileSync(`${DIR}/ref/feats.json`, "utf8"));
const manifest = JSON.parse(readFileSync(`${DIR}/ref/chunk0.json`, "utf8"));
const refBuf = readFileSync(`${DIR}/ref/chunk0.bin`);
const refBin = refBuf.buffer.slice(refBuf.byteOffset, refBuf.byteOffset + refBuf.byteLength);

function refView(name) {
  const m = manifest[name];
  if (!m) return null;
  const slice = refBin.slice(m.offset, m.offset + m.bytes);
  if (m.dtype === "float32") return { f: new Float32Array(slice), dtype: m.dtype, shape: m.shape };
  if (m.dtype === "bool" || m.dtype === "uint8") return { u8: new Uint8Array(slice), dtype: m.dtype, shape: m.shape };
  if (m.dtype === "int8") return { i8: new Int8Array(slice), dtype: m.dtype, shape: m.shape };
  if (m.dtype === "int32") return { i32: new Int32Array(slice), dtype: m.dtype, shape: m.shape };
  if (m.dtype === "int64") return { i64: new BigInt64Array(slice), dtype: m.dtype, shape: m.shape };
  return null;
}
function cmpTensor(gpuBuf, ref) {
  const n = ref.shape.reduce((a, b) => a * b, 1);
  const u32 = new Uint32Array(gpuBuf, 0, gpuBuf.byteLength / 4);
  let nBad = 0, maxAbs = 0, firstIdx = -1, got = 0, want = 0;
  if (ref.dtype === "float32") {
    const g = new Float32Array(gpuBuf, 0, n);
    for (let i = 0; i < n; i++) {
      const a = g[i], b = ref.f[i];
      if (Number.isNaN(a) && Number.isNaN(b)) continue;
      const d = Math.abs(a - b);
      if (!(Number.isNaN(a) === Number.isNaN(b)) || d > tolOf(b)) {
        if (firstIdx < 0) { firstIdx = i; got = a; want = b; }
        nBad++;
      }
      if (d > maxAbs) maxAbs = d;
    }
  } else {
    for (let i = 0; i < n; i++) {
      const a = u32[i] | 0;
      const b = ref.dtype === "bool" ? (ref.u8[i] ? 1 : 0) : ref.dtype === "int8" ? ref.i8[i] : ref.dtype === "int32" ? ref.i32[i] : Number(ref.i64[i]);
      if (a !== b) { if (firstIdx < 0) { firstIdx = i; got = a; want = b; } nBad++; }
    }
  }
  return { nBad, maxAbs, firstIdx, got, want, n };
}

console.log(`kernels=${doc.kernels.length} mode=${MODE}`);
const rt = new Runtime(doc, wBytes);
rt.layout(); rt.expand();
console.log(`arena ${(rt.aU32 * 4 / 1e6).toFixed(1)}MB dispatches=${rt.schedule.length}`);
const gpu = (globalThis.navigator ?? {}).gpu;
if (!gpu) { console.log("RESULT_JSON:" + JSON.stringify({ pass: false, error: "no navigator.gpu" })); Deno.exit(3); }
await rt.init(gpu);
console.log("gpu init ok");

const stateInputs = doc.meta.inputs.filter((n) => n !== "x");
{
  const flat = new Float32Array(featsJ[0].flat(Infinity).map(Number));
  rt.writeTensor("x", flat);
  for (const s of stateInputs) rt.zeroTensor(s);
}
let nPass = 0, worst = 0, worstT = "";
const t0 = performance.now();
for (let gi = 0; gi < rt.groups.length; gi++) {
  const g = rt.groups[gi];
  rt.runThrough(gi + 1);
  for (const outName of g.kernel.outputs) {
    const ref = refView(outName);
    if (!ref) continue;
    const gpuOut = await rt.readTensor(outName);
    const c = cmpTensor(gpuOut, ref);
    if (c.maxAbs > worst) { worst = c.maxAbs; worstT = outName; }
    if (c.nBad > 0) {
      const inRep = [];
      for (const inName of g.kernel.inputs) {
        const iref = refView(inName);
        if (!iref || !rt.A.has(inName)) { inRep.push([inName, "noref"]); continue; }
        const ic = cmpTensor(await rt.readTensor(inName), iref);
        inRep.push([inName, ic.nBad === 0 ? "OK" : `BAD ${ic.nBad}/${ic.n} got=${ic.got} want=${ic.want}`]);
      }
      console.log("RESULT_JSON:" + JSON.stringify({
        pass: false, gi, kind: g.kernel.kind, op: g.kernel.op, tensor: outName,
        nBad: c.nBad, n: c.n, got: c.got, want: c.want, inputs: inRep,
        nPass, elapsedMs: Math.round(performance.now() - t0),
      }));
      Deno.exit(3);
    }
  }
  nPass++;
  if (gi % 100 === 0) console.log(`  ...${gi}/${rt.groups.length} worst=${worst.toExponential(2)}@${worstT.slice(0, 30)}`);
}
console.log("RESULT_JSON:" + JSON.stringify({ pass: true, nPass, worstMaxAbs: worst, worstTensor: worstT, elapsedMs: Math.round(performance.now() - t0) }));
