// sim56.mjs — CPU VM that replays rt.js dispatch words exactly as the WGSL
// shaders would: same uniform layout, same strides, same arena reads.
// Usage: node sim56.mjs <groupIdx> <kernelsDir>
import { Runtime } from "./rt.js";
import { readFileSync } from "node:fs";
import { ELEM_OP, OP } from "./wgsl.js";

const DIR = process.argv[3] || "../out";
const GI = parseInt(process.argv[2] || "56");
const doc = JSON.parse(readFileSync(`${DIR}/kernels.json`, "utf8"));
const wBytes = readFileSync(`${DIR}/weights.bin`);
const man = JSON.parse(readFileSync("ref/chunk0.json", "utf8"));
const refBuf = readFileSync("ref/chunk0.bin");
const refBin = refBuf.buffer.slice(refBuf.byteOffset, refBuf.byteOffset + refBuf.byteLength);

const rt = new Runtime(doc, wBytes);
rt.layout(); rt.expand();

// arenas (u32 view + f32 alias)
const A = new Uint32Array(rt.aU32);
const F = new Float32Array(A.buffer);
const W = new Uint32Array(Math.ceil(wBytes.byteLength / 4));
{ // load weights.bin into the u32 view (same as rt.init writeBuffer)
  const u8 = wBytes.buffer.slice(wBytes.byteOffset, wBytes.byteOffset + wBytes.byteLength);
  W.set(new Uint32Array(u8));
}

function refArr(name) {
  const m = man[name];
  if (!m) return null;
  const dv = new DataView(refBin, m.offset, m.bytes);
  const n = m.shape.reduce((a, b) => a * b, 1);
  if (m.dtype === "float32") {
    const a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = dv.getFloat32(i * 4, true);
    return a;
  }
  if (m.dtype === "bool" || m.dtype === "uint8") {
    return new Uint8Array(refBin.slice(m.offset, m.offset + m.bytes));
  }
  if (m.dtype === "int64") {
    const a = new BigInt64Array(n);
    for (let i = 0; i < n; i++) a[i] = dv.getBigInt64(i * 8, true);
    return a;
  }
  if (m.dtype === "int32") {
    const a = new Int32Array(n);
    for (let i = 0; i < n; i++) a[i] = dv.getInt32(i * 4, true);
    return a;
  }
  if (m.dtype === "int8") return new Int8Array(refBin.slice(m.offset, m.offset + m.bytes));
  throw new Error("ref dtype " + m.dtype);
}
// seed: graph inputs (x + zero states) and every available ref tensor BEFORE
// the target group, so the VM starts from true upstream values
for (const name of Object.keys(man)) { /* seeded below per-group */ }

const group = rt.groups[GI];
const wordsOf = (d) => d.words;
const f32b = (u) => { const b = new ArrayBuffer(4); new Uint32Array(b)[0] = u; return new Float32Array(b)[0]; };

function coords(flat, shape4) {
  const c = [0, 0, 0, 0];
  let rem = flat;
  for (let r = 3; r >= 0; r--) { c[r] = rem % shape4[r]; rem = (rem / shape4[r]) | 0; }
  return c;
}
function ldIn(w, i, flat) {
  const off = [w[12], w[13], w[14]][i];
  const arena = ((w[15] >> i) & 1) ? W : A;
  return arena[off + flat];
}
function ldInF(w, i, flat) { return f32b(ldIn(w, i, flat)); }

// seed all tensors available in ref EXCEPT the target group's outputs
for (const [name, m] of Object.entries(man)) {
  const slot = rt.A.get(name);
  if (slot === undefined) continue;
  const arr = refArr(name);
  const n = m.shape.reduce((a, b) => a * b, 1);
  if (m.dtype === "float32") { for (let i = 0; i < n; i++) A[slot + i] = 0, F[slot + i] = arr[i]; }
  else if (m.dtype === "bool") { for (let i = 0; i < n; i++) A[slot + i] = arr[i] !== 0 ? 1 : 0; }
  else if (m.dtype === "int64") { for (let i = 0; i < n; i++) { const v = Number(arr[i]); A[slot + 2 * i] = v; A[slot + 2 * i + 1] = 0; } }
  else if (m.dtype === "int32" || m.dtype === "int8") { for (let i = 0; i < n; i++) A[slot + i] = arr[i] | 0; }
}

console.log(`simulating group #${GI} ${group.kernel.kind}[${group.kernel.op}] — ${group.dispatches.length} dispatches`);
const steps = group.kernel.kind === "fuseq" ? group.kernel.seq
  : group.dispatches.map(() => ({ outputs: group.kernel.outputs, op: group.kernel.op }));
let stepI = 0;
function stepCompare(d) {
  const st = steps[stepI++];
  if (!st) return;
  for (const oname of st.outputs) {
    const ref = refArr(oname);
    if (!ref) continue;
    const slot = rt.A.get(oname);
    let bad = 0, worst = 0, g0 = 0, w0 = 0;
    for (let i = 0; i < ref.length; i++) {
      const g = F[slot + i], wv = ref[i];
      const dd = Math.abs(g - wv);
      if (dd > 1e-4) { bad++; if (!worst || dd > worst) { worst = dd; g0 = g; w0 = wv; } }
    }
    console.log(`    -> ${oname.slice(0, 44)} nBad=${bad}/${ref.length} worst=${worst.toExponential(2)} got=${g0.toFixed(4)} want=${w0.toFixed(4)}`);
  }
}
for (const d of group.dispatches) {
  const w = wordsOf(d);
  const nOut = w[1];
  const outShape = w.slice(4, 8);
  if (d.pipe === "elem" || d.pipe === "copy" || d.pipe === "gather") {
    const op = w[44], intMode = w[45], sm = w[46] || 1;
    let bad = 0, firstGot = 0, firstWant = 0, nan = 0;
    for (let g = 0; g < nOut; g++) {
      const c = coords(g, outShape);
      let v;
      if (d.pipe === "copy") {
        v = ldIn(w, 0, g);
      } else if (d.pipe === "gather") {
        const gSt = w.slice(44, 48);       // op0: source strides
        const gOff = w[56];                // op3.x: source offset
        let src = gOff + c[0] * gSt[0] + c[1] * gSt[1] + c[2] * gSt[2] + c[3] * gSt[3];
        const dst = w[40] + c[0] * w[52] + c[1] * w[53] + c[2] * w[54] + c[3] * w[55];
        A[dst] = ldIn(w, 0, src);
        continue;
      } else {
        const off0 = c[0] * w[20] + c[1] * w[21] + c[2] * w[22] + c[3] * w[23];
        if (intMode) {
          const a = ldIn(w, 0, off0 * sm);
          const b = w[3] > 1 ? ldIn(w, 1, (c[0] * w[28] + c[1] * w[29] + c[2] * w[30] + c[3] * w[31]) * sm) : 0;
          if (op === OP.ADD) v = (a + b) | 0;
          else if (op === OP.SUB) v = (a - b) | 0;
          else if (op === OP.MUL) v = Math.imul(a, b);
          else v = a;
          A[w[40] + g * sm] = v;
          if (sm === 2) A[w[40] + g * 2 + 1] = 0;
          continue;
        }
        const fa = ldInF(w, 0, off0);
        let fv = fa;
        if (op === OP.WHERE) {
          const condOff = off0;
          const off1 = c[0] * w[28] + c[1] * w[29] + c[2] * w[30] + c[3] * w[31];
          const off2 = c[0] * w[36] + c[1] * w[37] + c[2] * w[38] + c[3] * w[39];
          const cond = ldIn(w, 0, condOff);
          const fx = ldInF(w, 1, off1);
          const fy = ldInF(w, 2, off2);
          const v2 = cond !== 0 ? fx : fy;
          A[w[40] + g] = 0; F[w[40] + g] = v2;
          continue;
        }
        if (w[3] > 1) {
          const off1 = c[0] * w[28] + c[1] * w[29] + c[2] * w[30] + c[3] * w[31];
          const fb = ldInF(w, 1, off1);
          if (op === OP.ADD) fv = fa + fb;
          else if (op === OP.SUB) fv = fa - fb;
          else if (op === OP.MUL) fv = fa * fb;
          else if (op === OP.DIV) fv = fa / fb;
          else if (op === OP.POW) fv = Math.pow(fa, fb);
        } else {
          if (op === OP.EXP) fv = Math.exp(fa);
          else if (op === OP.LOG) fv = Math.log(fa);
          else if (op === OP.SIGMOID) fv = 1 / (1 + Math.exp(-fa));
          else if (op === OP.TANH) fv = Math.tanh(fa);
          else if (op === OP.SQRT) fv = Math.sqrt(fa);
          else if (op === OP.RECIP) fv = 1 / fa;
          else if (op === OP.NEG) fv = -fa;
          else if (op === OP.ABS) fv = Math.abs(fa);
        }
        A[w[40] + g] = 0; F[w[40] + g] = fv;
        if (Number.isNaN(fv)) nan++;
      }
    }
    // step-level compare: kernel op name for this dispatch
    const stepOp = d.words ? Object.entries(ELEM_OP).find(([nm, cd]) => cd === op)?.[0] : "?";
    console.log(`  ${d.pipe} op=${op}(${stepOp}) n=${nOut} nan=${nan}`);
    stepCompare(d);
  } else if (d.pipe === "reduce") {
    const subop = w[44], nIn = w[45], scale = f32b(w[47]);
    const contrib = w.slice(60, 64);
    let nan = 0;
    for (let g = 0; g < nOut; g++) {
      let acc = 0;
      for (let e = 0; e < nIn; e++) {
        const c = coords(e, w.slice(16, 20));
        const of = c[0] * contrib[0] + c[1] * contrib[1] + c[2] * contrib[2] + c[3] * contrib[3];
        if (of === g) acc += ldInF(w, 0, e);
      }
      if (subop === 1) acc *= scale;
      F[w[40] + g] = acc;
      if (Number.isNaN(acc)) nan++;
    }
    console.log(`  reduce(${subop}) nOut=${nOut} nIn=${nIn} nan=${nan} contrib=${contrib}`);
    stepCompare(d);
  } else {
    console.log(`  ${d.pipe} n=${nOut} (not simulated — check separately if it matters)`);
  }
}
// compare group outputs
for (const outName of group.kernel.outputs) {
  const ref = refArr(outName);
  const slot = rt.A.get(outName);
  const n = ref.length;
  let bad = 0, nan = 0, worst = 0, got0 = 0, want0 = 0;
  for (let i = 0; i < n; i++) {
    const g = F[slot + i], wv = ref[i];
    if (Number.isNaN(g)) { nan++; if (!bad) { got0 = g; want0 = wv; } bad++; continue; }
    const dd = Math.abs(g - wv);
    if (dd > 1e-4 && !bad) { got0 = g; want0 = wv; }
    if (dd > 1e-4) bad++;
    if (dd > worst) worst = dd;
  }
  console.log(`OUT ${outName}: nBad=${bad}/${n} nan=${nan} worst=${worst.toExponential(2)} first got=${got0} want=${want0}`);
}
