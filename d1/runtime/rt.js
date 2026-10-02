// d1/runtime/rt.js — orchestrator: kernels.json + weights.bin -> WGSL dispatch
//
// Parity MVP: fuseq chains expand to per-op dispatches; matmul_int8_dq expands
// to qstat+qgemm; Concat expands to per-piece gathers. All ranks padded to 4
// (leading 1s) so shaders have a single coord decomposition path.
import { PIPE_SOURCES, ELEM_OP } from "./wgsl.js";

const BUF = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, UNIFORM: 0x40, STORAGE: 0x80 };
const SLOT = { float32: 1, bool: 1, int32: 1, uint8: 1, int8: 1, int64: 2 };

function fail(msg) { throw new Error("rt: " + msg); }
function prod(a) { return a.reduce((x, y) => x * y, 1); }
function pad4(shape) { return [1, 1, 1, 1].slice(0, 4 - shape.length).concat(shape.map(Number)); }
function strides4(shape4) {
  const s = [0, 0, 0, 0];
  let acc = 1;
  for (let r = 3; r >= 0; r--) { s[r] = acc; acc *= shape4[r] || 1; }
  return s;
}
function f32b(f) { const b = new Float32Array([f]); return new Uint32Array(b.buffer)[0]; }

export class Runtime {
  constructor(doc, weightsBytes) {
    this.doc = doc;
    // node fs gives Buffer; normalize to ArrayBuffer
    this.wBytes = weightsBytes instanceof ArrayBuffer ? weightsBytes
      : weightsBytes.buffer.slice(weightsBytes.byteOffset,
                                 weightsBytes.byteOffset + weightsBytes.byteLength);
    this.tensors = doc.tensors;
    this.wOf = new Map();
    for (const m of doc.weights) {
      this.wOf.set(m.name, { offU32: m.offset >> 2, dtype: m.dtype, shape: m.shape });
    }
    this.groups = [];
    this.A = new Map();
    this.scratchOf = new Map();
  }

  wVal(name) {
    const m = this.wOf.get(name);
    if (!m) fail("weight not packed: " + name);
    const dv = new DataView(this.wBytes, m.offU32 * 4);
    const n = prod(m.shape);
    const out = [];
    for (let i = 0; i < n; i++) {
      if (m.dtype === "int64") out.push(dv.getBigInt64(i * 8, true));
      else if (m.dtype === "int32") out.push(dv.getInt32(i * 4, true));
      else if (m.dtype === "float32") out.push(dv.getFloat32(i * 4, true));
      // 8-bit initializers are widened to u32/elem in weights.bin
      else if (m.dtype === "int8") out.push(dv.getInt32(i * 4, true));
      else if (m.dtype === "bool") out.push(dv.getUint32(i * 4, true));
      else fail("wVal dtype " + m.dtype);
    }
    return out;
  }

  shapeOf(name) {
    if (this.tensors[name]) return this.tensors[name].shape.map(Number);
    const w = this.wOf.get(name);
    if (w) return w.shape.map(Number);
    fail("no shape for tensor " + name);
  }
  dtypeOf(name) {
    if (this.tensors[name]) return this.tensors[name].dtype;
    const w = this.wOf.get(name);
    if (w) return w.dtype;
    fail("no dtype for tensor " + name);
  }

  // ---- arena layout FIRST (encode depends on it) ----
  layout() {
    let off = 0;
    const put = (name) => {
      if (this.A.has(name)) return;
      const dt = this.dtypeOf(name);
      const slots = SLOT[dt];
      if (!slots) fail("dtype " + dt + " for " + name);
      this.A.set(name, off);
      off += prod(this.shapeOf(name)) * slots;
    };
    for (const n of this.doc.meta.inputs) put(n);
    for (const n of this.doc.meta.outputs) put(n);
    for (const name of Object.keys(this.tensors)) put(name);
    let nMmi = 0;
    for (const k of this.doc.kernels) if (k.kind === "matmul_int8_dq") nMmi++;
    this.scratchBase = off;
    off += nMmi * 3;
    this.aU32 = off;
    return this;
  }
  scratchOfKernel(ki) {  // stable per original kernel index
    let n = 0;
    for (let i = 0; i < ki; i++) if (this.doc.kernels[i].kind === "matmul_int8_dq") n++;
    return this.scratchBase + n * 3;
  }

  expand() {
    const groups = [];
    this.doc.kernels.forEach((k, ki) => {
      const g = { kernel: k, dispatches: [], groupIdx: ki };
      if (k.kind === "fuseq") {
        for (const step of k.seq) {
          const kind = step.op in ELEM_OP ? "elem" : (step.op.startsWith("Reduce") ? "reduce" : null);
          if (!kind) fail("fuseq step op not supported: " + step.op);
          g.dispatches.push(this.encode({ kind, op: step.op, attrs: step.attrs,
            inputs: step.inputs, outputs: step.outputs }, ki));
        }
      } else if (k.kind === "matmul_int8_dq") {
        const pseudo = "@q" + ki;
        g.dispatches.push(this.encode({ kind: "qstat", op: "qstat", attrs: {},
          inputs: [k.inputs[0]], outputs: k.outputs,
          __side_scale: k.side_scale, __side_zp: k.side_zp }, ki));
        g.dispatches.push(this.encode({ kind: "qgemm", op: "qgemm", attrs: {},
          inputs: k.inputs, outputs: k.outputs,
          __scale_tensor: k.scale_tensor }, ki));
        // dependency wiring: qstat -> scratch pseudo -> qgemm; side outputs
        // feed external consumers; scale_tensor gates qgemm
        const q0 = g.dispatches[g.dispatches.length - 2];
        const q1 = g.dispatches[g.dispatches.length - 1];
        q0.deps = { in: [k.inputs[0]],
                    out: [k.side_scale, k.side_zp, pseudo].filter(Boolean) };
        q1.deps = { in: [...k.inputs, k.scale_tensor, pseudo].filter(Boolean),
                    out: [...k.outputs] };
      } else if (k.kind === "layout" && k.op === "Concat") {
        const rank = this.shapeOf(k.outputs[0]).length;
        const axis = (k.attrs.axis !== undefined ? k.attrs.axis : this.wVal(k.inputs[1])[0]);
        const ax = axis < 0 ? axis + rank : axis;
        let start = 0;
        for (const inp of k.inputs) {
          const sh = this.shapeOf(inp);
          g.dispatches.push(this.encode({ kind: "gather", op: "Concat",
            attrs: { __piece: { start, axis: ax } }, inputs: [inp],
            outputs: k.outputs }, ki, { pieceShape: sh, pieceRank: rank }));
          start += sh[ax];
        }
      } else {
        const map = {
          elementwise: "elem", control: "elem", reduce: "reduce", softmax: "softmax",
          matmul_fp: "matmul_fp", conv: "conv",
          layout: { Reshape: "copy", Squeeze: "copy", Unsqueeze: "copy",
                    Flatten: "copy", Transpose: "gather", Slice: "gather",
                    Expand: "gather", GatherElements: "gelem" }[k.op],
        };
        if (!map[k.kind]) fail("kind not supported: " + k.kind);
        g.dispatches.push(this.encode({ kind: map[k.kind], op: k.op,
          attrs: k.attrs, inputs: k.inputs, outputs: k.outputs }, ki));
      }
      for (const d of g.dispatches) {
        if (!d.deps) d.deps = { in: [...k.inputs], out: [...k.outputs] };
        d.groupIdx = ki;
      }
      groups.push(g);
    });
    this.groups = groups;
    this.topoSchedule();
    this.executed = 0;
    return this;
  }

  // ---- global dispatch scheduling (Kahn, stable) ----
  // Kernel list order from kgen is advisory; true data deps (incl. qstat
  // side outputs and lazy-scale tensors) decide execution order.
  topoSchedule() {
    const all = [];
    for (const g of this.groups) for (const d of g.dispatches) all.push(d);
    const weight = new Set(this.wOf.keys());
    const prodOf = new Map();  // tensor -> dispatch
    for (const d of all) for (const o of d.deps.out) prodOf.set(o, d);
    const indeg = new Map(all.map((d) => [d, 0]));
    const outs = new Map(all.map((d) => [d, []]));
    for (const d of all) {
      for (const t of d.deps.in) {
        if (weight.has(t)) continue;
        const p = prodOf.get(t);
        if (p && p !== d) {
          indeg.set(d, indeg.get(d) + 1);
          outs.get(p).push(d);
        }
      }
    }
    const ready = all.filter((d) => indeg.get(d) === 0);
    const order = [];
    while (ready.length) {
      const d = ready.shift();
      order.push(d);
      for (const c of outs.get(d)) {
        indeg.set(c, indeg.get(c) - 1);
        if (indeg.get(c) === 0) ready.push(c);
      }
    }
    if (order.length !== all.length) fail("dispatch schedule has a cycle");
    // flatten back into groups; each group keeps its dispatches in order
    const byGroup = new Map();
    for (const d of order) {
      if (!byGroup.has(d.groupIdx)) byGroup.set(d.groupIdx, []);
      byGroup.get(d.groupIdx).push(d);
    }
    this.schedule = order;
    this.groupEnd = new Array(this.groups.length).fill(0);
    for (let i = 0; i < order.length; i++) {
      order[i].__sched = i;
      this.groupEnd[order[i].groupIdx] = i + 1;
    }
  }

  runThroughSchedule(posIncl) {
    if (posIncl + 1 <= this.executed) { return; }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = this.executed; i <= posIncl; i++) {
      const d = this.schedule[i];
      pass.setPipeline(this.pipes[d.pipe]);
      pass.setBindGroup(0, this.bg, [d.block * 512]);
      pass.dispatchWorkgroups(d.wg);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.executed = posIncl + 1;
  }

  encode(k, ki, extra = {}) {
    const outName = k.outputs[0];
    const pieceShape = extra.pieceShape || this.shapeOf(outName);
    const outSh4 = pad4(pieceShape);
    const nOut = prod(pieceShape);
    const w = new Uint32Array(128).fill(0);
    const putIn = (i, name) => {
      const inW = this.wOf.has(name);
      const off = inW ? this.wOf.get(name).offU32
                       : (this.A.get(name) ?? fail("no arena slot: " + name));
      w[12 + i] = off;
      if (inW) w[15] |= (1 << i);
      const sh = pad4(this.shapeOf(name));
      const st = strides4(sh);
      for (let d = 0; d < 4; d++) if (sh[d] === 1 && outSh4[d] !== 1) st[d] = 0;
      w.set(sh, 16 + i * 8);
      w.set(st, 20 + i * 8);
      return { sh, st };
    };
    const setOut = () => {
      w[1] = nOut; w[2] = 4; w[3] = k.inputs.length;
      w.set(outSh4, 4);
      w.set(strides4(outSh4), 8);
      w[40] = this.A.get(outName) ?? fail("no arena slot out: " + outName);
    };
    const opWords = (idx, vals) => w.set(vals, 44 + idx * 4);

    if (k.kind === "elem") {
      const code = ELEM_OP[k.op];
      if (code === undefined) fail("elem op " + k.op);
      const INT_DT = new Set(["int64", "int32", "int8", "uint8", "bool"]);
      const outDt = this.dtypeOf(outName);
      const intMode = INT_DT.has(outDt)
        || k.inputs.some((n) => INT_DT.has(this.dtypeOf(n)));
      const slotMul = SLOT[outDt] || 1;
      if (intMode && slotMul === 2
          && !k.inputs.every((n) => this.dtypeOf(n) === "int64")) {
        fail("mixed-dtype int64 elem op: " + outName);
      }
      k.inputs.forEach((n, i) => putIn(i, n));
      setOut();
      opWords(0, [code, intMode ? 1 : 0, slotMul, 0]);
      return { pipe: "elem", nOut, words: w, wg: Math.ceil(nOut / 64) };
    }
    if (k.kind === "gather") {
      const in0 = k.inputs[0];
      const { st } = putIn(0, in0);
      setOut();
      let gOff = 0;
      const g = [...st];
      if (k.op === "Transpose") {
        const perm = k.attrs.perm;
        const rank = perm.length;
        const gg = [0, 0, 0, 0];
        for (let r = 0; r < rank; r++) gg[4 - rank + r] = st[4 - rank + perm[r]];
        g.splice(0, 4, ...gg);
      } else if (k.op === "Slice") {
        const starts = this.wVal(k.inputs[1]);
        const ends = this.wVal(k.inputs[2]);
        const axes = k.inputs[3] ? this.wVal(k.inputs[3]) : starts.map((_, i) => i);
        const steps = k.inputs[4] ? this.wVal(k.inputs[4]) : starts.map(() => 1);
        const inSh = this.shapeOf(in0);
        const rank = inSh.length;
        for (let a = 0; a < axes.length; a++) {
          const step = Number(steps[a]);
          const ax = Number(axes[a]) < 0 ? Number(axes[a]) + rank : Number(axes[a]);
          const r = inSh[ax];
          const stAx = st[4 - rank + ax];
          let len;
          if (step > 0) {
            const s = Number(starts[a]) < 0 ? Number(starts[a]) + r : Number(starts[a]);
            const e = Number(ends[a]) < 0 ? Number(ends[a]) + r : Math.min(Number(ends[a]), r);
            len = Math.ceil((e - s) / step);
            gOff += s * stAx;
            if (step !== 1) g[4 - rank + ax] = step * stAx;
          } else {
            // negative step (u32 two's-complement wrap in the shader is exact)
            const s = Number(starts[a]) < 0 ? Number(starts[a]) + r
                      : Math.min(Number(starts[a]), r - 1);
            const e = Number(ends[a]) < -r ? -1
                      : (Number(ends[a]) < 0 ? Number(ends[a]) + r
                         : Math.min(Number(ends[a]), r - 1));
            len = Math.ceil((s - e) / -step);
            gOff += s * stAx;
            g[4 - rank + ax] = (step * stAx) >>> 0;
          }
          if (len < 0 || len !== pieceShape[ax]) {
            fail(`slice len mismatch ${outName}: ${len} vs ${pieceShape[ax]}`);
          }
        }
      } else if (k.op === "Concat") {
        const p = k.attrs.__piece;
        const inSt = strides4(pad4(this.shapeOf(in0)));
        g.splice(0, 4, ...inSt);
        w[1] = prod(pieceShape);
        // piece offset lives on the OUTPUT side: out[..., start+i] = in[..., i]
        const outStReal = strides4(pad4(this.shapeOf(k.outputs[0])));
        w[40] += p.start * outStReal[4 - extra.pieceRank + p.axis];
      } else if (k.op === "Expand") {
        // broadcast copy: st already zeroed for bcast dims by putIn
      } else fail("gather op " + k.op);
      if ((SLOT[this.dtypeOf(outName)] || 1) !== 1) {
        fail("gather on multi-slot dtype: " + this.dtypeOf(outName));
      }
      opWords(0, g);
      // destination strides: full-tensor strides for concat pieces (the
      // piece scatters when the concat axis is not innermost)
      const dstSt = k.op === "Concat"
        ? strides4(pad4(this.shapeOf(k.outputs[0])))
        : strides4(outSh4);
      opWords(2, dstSt);
      opWords(3, [gOff, 0, 0, 0]);
      return { pipe: "gather", nOut: w[1], words: w, wg: Math.ceil(w[1] / 64) };
    }
    if (k.kind === "copy") {
      // Reshape/Squeeze/Unsqueeze with static shapes: FLAT copy (a coordinate
      // gather breaks when source and target ranks/shapes differ, e.g.
      // [1,32,128,19] -> [1,32,2432])
      putIn(0, k.inputs[0]); setOut();
      const units = nOut * (SLOT[this.dtypeOf(outName)] || 1);
      w[1] = units;
      return { pipe: "copy", nOut: units, words: w, wg: Math.ceil(units / 64) };
    }
    if (k.kind === "gelem") {
      putIn(0, k.inputs[0]); putIn(1, k.inputs[1]); setOut();
      if (this.dtypeOf(k.inputs[1]) === "int64") {
        const sh = pad4(this.shapeOf(k.inputs[1]));
        const st = strides4(sh).map((s) => s * 2);
        w.set(sh, 24); w.set(st, 28);
      }
      const rank = this.shapeOf(outName).length;
      const ax = k.attrs.axis < 0 ? k.attrs.axis + rank : k.attrs.axis;
      opWords(0, [4 - rank + ax, 0, 0, 0]);
      return { pipe: "gelem", nOut, words: w, wg: Math.ceil(nOut / 64) };
    }
    if (k.kind === "reduce") {
      const in0 = k.inputs[0];
      const inSh = this.shapeOf(in0);
      const rank = inSh.length;
      const outSh = this.shapeOf(outName);
      const outRank = outSh.length;
      let axes = k.attrs.axes ? [...k.attrs.axes]
        : (k.inputs[1] ? this.wVal(k.inputs[1]).map(Number) : inSh.map((_, i) => i));
      axes = axes.map((a) => (a < 0 ? a + rank : a));
      const redSet = new Set(axes);
      const outSt4 = strides4(pad4(outSh));
      // per padded input axis: output stride contribution (0 = reduced axis)
      const contrib = [0, 0, 0, 0];
      if (k.attrs.keepdims === 0) {
        let oa = 0;
        for (let r = 0; r < rank; r++) {
          if (redSet.has(r)) continue;
          contrib[4 - rank + r] = outSt4[4 - outRank + oa];
          oa++;
        }
        if (oa !== outRank) fail(`reduce shape mapping ${outName}`);
      } else {
        if (outRank !== rank) fail(`reduce keepdims rank ${outName}`);
        for (let r = 0; r < rank; r++) {
          if (!redSet.has(r)) contrib[4 - rank + r] = outSt4[4 - rank + r];
        }
      }
      const nIn = prod(inSh);
      const nRed = Math.round(nIn / nOut);
      if (nRed * nOut !== nIn) fail("reduce ratio " + outName);
      putIn(0, in0); setOut();
      opWords(0, [k.op === "ReduceMean" ? 1 : 0, nIn, 0, f32b(1 / nRed)]);
      opWords(4, contrib);
      return { pipe: "reduce", nOut, words: w, wg: Math.ceil(nOut / 64) };
    }
    if (k.kind === "softmax") {
      putIn(0, k.inputs[0]); setOut();
      const rank = this.shapeOf(outName).length;
      const axRaw = k.attrs.axis ?? -1;
      const ax = axRaw < 0 ? axRaw + rank : axRaw;
      const rowLen = this.shapeOf(k.inputs[0])[ax];
      opWords(0, [4 - rank + ax, k.op === "LogSoftmax" ? 1 : 0, 0, 0]);
      const nRows = nOut / rowLen;
      if (!Number.isInteger(nRows)) fail("softmax rows");
      return { pipe: "softmax", nOut, words: w, wg: nRows };
    }
    if (k.kind === "matmul_fp") {
      const aSh = this.shapeOf(k.inputs[0]);
      const bSh = this.shapeOf(k.inputs[1]);
      const M = aSh[aSh.length - 2], K = aSh[aSh.length - 1], N = bSh[bSh.length - 1];
      if (K !== bSh[bSh.length - 2]) fail("matmul K mismatch " + outName);
      putIn(0, k.inputs[0]); putIn(1, k.inputs[1]); setOut();
      const bBatch = prod(bSh.slice(0, -2));
      opWords(0, [M, N, K, prod(aSh.slice(0, -2))]);
      opWords(1, [bBatch > 1 ? K * N : 0, 0, 0, 0]);
      return { pipe: "matmul_fp", nOut, words: w, wg: Math.ceil(nOut / 64) };
    }
    if (k.kind === "qstat") {
      const n = prod(this.shapeOf(k.inputs[0]));
      putIn(0, k.inputs[0]);
      w[1] = n; w[2] = 4; w[3] = 1;
      w[40] = this.scratchOfKernel(ki);
      const sideS = k.__side_scale ? (this.A.get(k.__side_scale) ?? fail("no slot " + k.__side_scale)) : 0;
      const sideZ = k.__side_zp ? (this.A.get(k.__side_zp) ?? fail("no slot " + k.__side_zp)) : 0;
      opWords(2, [this.scratchOfKernel(ki), sideS, sideZ, 0]);
      return { pipe: "qstat", nOut: 1, words: w, wg: 1 };
    }
    if (k.kind === "qgemm") {
      const xSh = this.shapeOf(k.inputs[0]);
      const M = xSh[xSh.length - 2], K = xSh[xSh.length - 1];
      const wEnt = this.wOf.get(k.inputs[1]);
      if (!wEnt) fail("qgemm weight not in arena: " + k.inputs[1]);
      const N = wEnt.shape[1];
      let wScale = 1.0, bZp = 0;
      for (const ex of k.inputs.slice(2)) {
        const dt = this.dtypeOf(ex);
        if (dt === "float32") wScale = this.wVal(ex)[0];
        else if (dt === "int8") bZp = this.wVal(ex)[0];
        else fail("qgemm extra dtype " + dt);
      }
      putIn(0, k.inputs[0]);            // slot 0: x in A
      w[13] = wEnt.offU32; w[15] |= 2;  // slot 1: int8 W in W-arena
      setOut();
      opWords(0, [M, N, K, 0]);
      opWords(1, [f32b(wScale), bZp, 0, 0]);
      const stSlot = k.__scale_tensor ? (this.A.get(k.__scale_tensor) ?? fail("no slot " + k.__scale_tensor)) : 0;
      opWords(2, [this.scratchOfKernel(ki), 0, 0, stSlot]);
      return { pipe: "qgemm", nOut, words: w, wg: Math.ceil(nOut / 64) };
    }
    if (k.kind === "conv") {
      let inSh = this.shapeOf(k.inputs[0]);
      const wEnt = this.wOf.get(k.inputs[1]);
      if (!wEnt) fail("conv weight not arena: " + k.inputs[1]);
      const group = k.attrs.group ?? 1;
      let ws = wEnt.shape.map(Number);
      let padTop = 0, padLeft = 0;
      const pads = k.attrs.pads ?? [];
      if (inSh.length === 3) {
        // 1D conv [N,C,T] -> virtual 2D with H=1; pads [begin,end] on time
        if (pads.length !== 2) fail("1D conv pads len " + pads.length);
        padLeft = pads[0];
        inSh = [inSh[0], inSh[1], 1, inSh[2]];
        ws = [ws[0], ws[1], 1, ws[2]];
      } else if (inSh.length === 4) {
        padTop = pads[0] ?? 0;
        padLeft = pads[1] ?? 0;
      } else fail("conv rank " + inSh.length);
      const [Cout, CinG, KH, KW] = ws;
      const [, Cin, H, W] = inSh;
      if (Cin !== CinG * group) fail("conv group mismatch " + outName);
      const outSh = this.shapeOf(outName);
      const [, , H_out, W_out] = outSh.length === 3 ? [0, 0, 1, outSh[2]] : outSh;
      putIn(0, k.inputs[0]); putIn(1, k.inputs[1]);
      if (k.inputs[2]) putIn(2, k.inputs[2]);
      setOut();
      opWords(0, [Cout, CinG, KH, KW]);
      opWords(1, [H, W, H_out, W_out]);
      opWords(2, [k.attrs.strides?.[0] ?? 1, k.attrs.strides?.[1] ?? 1,
                  k.attrs.dilations?.[0] ?? 1, k.attrs.dilations?.[1] ?? 1]);
      opWords(3, [padTop, padLeft, 0, 0]);
      w.set([group, 0, 0, 0], 32);  // in2Shape.x = group (shader reads it)
      return { pipe: "conv", nOut, words: w, wg: Math.ceil(nOut / 64) };
    }
    fail("encode: unhandled " + k.kind + " " + (k.op || ""));
  }

  // ---- GPU ----
  async init(navigator) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) fail("no adapter");
    this.device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const dev = this.device;
    this.wBuf = dev.createBuffer({ size: Math.ceil(this.wBytes.byteLength / 4) * 4,
      usage: BUF.STORAGE | BUF.COPY_DST });
    dev.queue.writeBuffer(this.wBuf, 0, this.wBytes);
    this.aBuf = dev.createBuffer({ size: this.aU32 * 4,
      usage: BUF.STORAGE | BUF.COPY_SRC | BUF.COPY_DST });
    const nBlocks = this.groups.reduce((s, g) => s + g.dispatches.length, 0);
    this.nBlocks = nBlocks;
    this.uni = dev.createBuffer({ size: nBlocks * 512, usage: BUF.UNIFORM | BUF.COPY_DST });
    const all = new Uint32Array(nBlocks * 128);
    let bi = 0;
    for (const g of this.groups) for (const d of g.dispatches) {
      all.set(d.words, bi * 128); d.block = bi++;
    }
    dev.queue.writeBuffer(this.uni, 0, all);

    // explicit shared bind group layout (layout:"auto" is per-pipeline)
    const bgl = dev.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 512 } },
      ],
    });
    this.bg = dev.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.wBuf } },
        { binding: 1, resource: { buffer: this.aBuf } },
        { binding: 2, resource: { buffer: this.uni, size: 512 } },
      ],
    });
    const pl = dev.createPipelineLayout({ bindGroupLayouts: [bgl] });
    this.pipes = {};
    for (const [name, src] of Object.entries(PIPE_SOURCES)) {
      const sm = dev.createShaderModule({ code: src });
      const info = await sm.getCompilationInfo();
      const errs = info.messages.filter((m) => m.type === "error");
      if (errs.length) fail(`shader ${name} line ${errs[0].lineNum}: ${errs[0].message}`);
      this.pipes[name] = dev.createComputePipeline({ layout: pl,
        compute: { module: sm, entryPoint: "main" } });
    }
    return this;
  }

  writeTensor(name, typedArray) {
    const off = this.A.get(name) ?? fail("writeTensor no slot " + name);
    this.device.queue.writeBuffer(this.aBuf, off * 4,
      typedArray.buffer, typedArray.byteOffset, typedArray.byteLength);
  }
  zeroTensor(name) {
    const n = prod(this.shapeOf(name)) * (SLOT[this.dtypeOf(name)] || 1);
    this.writeTensor(name, new Uint32Array(n));
  }

  // incremental execution: dispatches are pure functions of arena state,
  // so a runThrough(giEnd) only submits dispatches not yet executed
  runThrough(giEnd) {
    const cut = giEnd >= this.groupEnd.length ? this.schedule.length
      : this.groupEnd[giEnd];
    if (cut <= this.executed) { return; }
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let i = this.executed; i < cut; i++) {
      const d = this.schedule[i];
      pass.setPipeline(this.pipes[d.pipe]);
      pass.setBindGroup(0, this.bg, [d.block * 512]);
      pass.dispatchWorkgroups(d.wg);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.executed = cut;
  }
  resetExecution() { this.executed = 0; }

  async readRaw(offU32, nWords) {
    const staging = this.device.createBuffer({ size: nWords * 4, usage: BUF.MAP_READ | BUF.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.aBuf, offU32 * 4, staging, 0, nWords * 4);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    return copy;
  }

  async readTensor(name) {
    const off = this.A.get(name) ?? fail("read no slot " + name);
    const dt = this.dtypeOf(name);
    const n = prod(this.shapeOf(name));
    const bytes = dt === "int64" ? n * 8 : n * 4;
    const staging = this.device.createBuffer({ size: bytes,
      usage: BUF.MAP_READ | BUF.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.aBuf, off * 4, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const copy = staging.getMappedRange().slice(0);
    staging.unmap(); staging.destroy();
    return copy;
  }
}
