// d1/runtime/wgsl.js — WGSL pipeline sources for the kernels.json runtime.
//
// Conventions shared by every pipeline:
//   * arenas W (weights, read-only) + A (intermediates, read_write), u32 view;
//     f32 values bitcast on load/store; bool/int32 stored as u32 slots
//   * one bind group: {W, A, uniform-with-dynamic-offset}; params block 512B
//   * params word map (see rt.js writer — keep in sync):
//       0 opcode | 1 nOut | 2 rankOut | 3 inCount
//       4-7   outShape (dims; coords() walks rankOut dims, row-major)
//       8-11  outStride (element strides)
//       12-14 inOff[3] (element offsets; arena selected by srcW bit i)
//       15    srcW bitmask
//       16-19 in0Shape | 20-23 in0Stride  (left-padded to rankOut, stride0=bcast)
//       24-27 in1Shape | 28-31 in1Stride
//       32-35 in2Shape | 36-39 in2Stride
//       40 outOff | 41-43 pad
//       44-47 op0 ... 76-79 op8 (per-pipeline semantics below)

export const OP = {
  ADD: 0, SUB: 1, MUL: 2, DIV: 3, EXP: 4, LOG: 5, POW: 6, SIGMOID: 7,
  TANH: 8, NEG: 9, ABS: 10, MAX: 11, MIN: 12, EQUAL: 13, LESS: 14,
  LESSOREQUAL: 15, GREATER: 16, WHERE: 17, SQRT: 18, RECIP: 19,
};
export const ELEM_OP = {
  Add: OP.ADD, Sub: OP.SUB, Mul: OP.MUL, Div: OP.DIV, Exp: OP.EXP,
  Log: OP.LOG, Pow: OP.POW, Sigmoid: OP.SIGMOID, Tanh: OP.TANH, Neg: OP.NEG,
  Abs: OP.ABS, Max: OP.MAX, Min: OP.MIN, Equal: OP.EQUAL, Less: OP.LESS,
  LessOrEqual: OP.LESSOREQUAL, Greater: OP.GREATER, Where: OP.WHERE,
  Sqrt: OP.SQRT, Reciprocal: OP.RECIP,
};

const PRELUDE = /* wgsl */`
struct Params {
  opcode: u32, nOut: u32, rankOut: u32, inCount: u32,
  outShape: vec4<u32>,
  outStride: vec4<u32>,
  inOff: vec3<u32>,
  srcW: u32,
  in0Shape: vec4<u32>, in0Stride: vec4<u32>,
  in1Shape: vec4<u32>, in1Stride: vec4<u32>,
  in2Shape: vec4<u32>, in2Stride: vec4<u32>,
  outOff: u32, pad0: u32, pad1: u32, pad2: u32,
  op0: vec4<u32>, op1: vec4<u32>, op2: vec4<u32>, op3: vec4<u32>,
  op4: vec4<u32>, op5: vec4<u32>, op6: vec4<u32>, op7: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read_write> A: array<u32>;
@group(0) @binding(2) var<uniform> P: Params;

fn ldfA(i: u32) -> f32 { return bitcast<f32>(A[i]); }
fn stfA(i: u32, v: f32) { A[i] = bitcast<u32>(v); }
fn rhte(x: f32) -> f32 {  // ONNX round-half-to-even
  let f = floor(x); let r = x - f;
  if (r > 0.5) { return f + 1.0; }
  if (r < 0.5) { return f; }
  return select(f, f + 1.0, (f % 2.0) != 0.0);
}
fn sig(x: f32) -> f32 { return 1.0 / (1.0 + exp(-x)); }
fn inArenaW(i: u32) -> bool { return ((P.srcW >> i) & 1u) == 1u; }
fn ldIn(i: u32, flat: u32) -> f32 {
  if (i == 0u) {
    if (inArenaW(0u)) { return bitcast<f32>(W[P.inOff.x + flat]); }
    return bitcast<f32>(A[P.inOff.x + flat]);
  } else if (i == 1u) {
    if (inArenaW(1u)) { return bitcast<f32>(W[P.inOff.y + flat]); }
    return bitcast<f32>(A[P.inOff.y + flat]);
  }
  if (inArenaW(2u)) { return bitcast<f32>(W[P.inOff.z + flat]); }
  return bitcast<f32>(A[P.inOff.z + flat]);
}
fn ldInU(i: u32, flat: u32) -> u32 {
  if (i == 0u) {
    if (inArenaW(0u)) { return W[P.inOff.x + flat]; }
    return A[P.inOff.x + flat];
  } else if (i == 1u) {
    if (inArenaW(1u)) { return W[P.inOff.y + flat]; }
    return A[P.inOff.y + flat];
  }
  if (inArenaW(2u)) { return W[P.inOff.z + flat]; }
  return A[P.inOff.z + flat];
}
fn coords(flat: u32) -> vec4<u32> {
  var c = vec4<u32>(0u);
  var rem = flat;
  for (var r: i32 = i32(P.rankOut) - 1; r >= 0; r--) {
    let d = P.outShape[u32(r)];
    c[u32(r)] = rem % d;
    rem = rem / d;
  }
  return c;
}
fn offOf(strides: vec4<u32>, c: vec4<u32>) -> u32 {
  var o = 0u;
  for (var r: u32 = 0u; r < P.rankOut; r++) { o += c[r] * strides[r]; }
  return o;
}
`;

// ---- elementwise (broadcast, 1-3 inputs, bool out for comparisons) ----
// op0.x = subop
const ELEM = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  let c = coords(g.x);
  let o = P.op0.x;
  if (o == ${OP.WHERE}) {
    let cond = ldInU(0u, offOf(P.in0Stride, c));
    let x = ldIn(1u, offOf(P.in1Stride, c));
    let y = ldIn(2u, offOf(P.in2Stride, c));
    stfA(P.outOff + g.x, select(y, x, cond != 0u));
    return;
  }
  if (P.op0.y == 1u) {
    // integer path: raw u32/i32 semantics (values fit low word; denorm-FTZ
    // makes f32 bitcast arithmetic unusable on GPUs)
    let a = ldInU(0u, offOf(P.in0Stride, c));
    let b = select(0u, ldInU(1u, offOf(P.in1Stride, c)), P.inCount > 1u);
    var r = a;
    if (o == ${OP.ADD}) { r = a + b; }
    else if (o == ${OP.SUB}) { r = a - b; }
    else if (o == ${OP.MUL}) { r = a * b; }
    else if (o == ${OP.MAX}) { r = select(b, a, i32(a) > i32(b)); }
    else if (o == ${OP.MIN}) { r = select(b, a, i32(a) < i32(b)); }
    else if (o == ${OP.EQUAL} || o == ${OP.LESS} || o == ${OP.LESSOREQUAL} || o == ${OP.GREATER}) {
      var bb = false;
      if (o == ${OP.EQUAL}) { bb = a == b; }
      else if (o == ${OP.LESS}) { bb = i32(a) < i32(b); }
      else if (o == ${OP.LESSOREQUAL}) { bb = i32(a) <= i32(b); }
      else { bb = i32(a) > i32(b); }
      A[P.outOff + g.x] = select(0u, 1u, bb);
      return;
    }
    A[P.outOff + g.x] = r;
    return;
  }
  var v = ldIn(0u, offOf(P.in0Stride, c));
  if (o == ${OP.EQUAL} || o == ${OP.LESS} || o == ${OP.LESSOREQUAL} || o == ${OP.GREATER}) {
    let w = ldIn(1u, offOf(P.in1Stride, c));
    var b = false;
    if (o == ${OP.EQUAL}) { b = v == w; }
    else if (o == ${OP.LESS}) { b = v < w; }
    else if (o == ${OP.LESSOREQUAL}) { b = v <= w; }
    else { b = v > w; }
    A[P.outOff + g.x] = select(0u, 1u, b);
    return;
  }
  if (P.inCount > 1u) {
    let w = ldIn(1u, offOf(P.in1Stride, c));
    if (o == ${OP.ADD}) { v = v + w; }
    else if (o == ${OP.SUB}) { v = v - w; }
    else if (o == ${OP.MUL}) { v = v * w; }
    else if (o == ${OP.DIV}) { v = v / w; }
    else if (o == ${OP.POW}) { v = pow(v, w); }
    else if (o == ${OP.MAX}) { v = max(v, w); }
    else if (o == ${OP.MIN}) { v = min(v, w); }
    else { }
  }
  if (o == ${OP.EXP}) { v = exp(v); }
  else if (o == ${OP.LOG}) { v = log(v); }
  else if (o == ${OP.SIGMOID}) { v = sig(v); }
  else if (o == ${OP.TANH}) { v = tanh(v); }
  else if (o == ${OP.NEG}) { v = -v; }
  else if (o == ${OP.ABS}) { v = abs(v); }
  else if (o == ${OP.SQRT}) { v = sqrt(v); }
  else if (o == ${OP.RECIP}) { v = 1.0 / v; }
  stfA(P.outOff + g.x, v);
}
`;

// ---- generic strided gather (Transpose / Slice / Concat piece / Expand) ----
// src = inOff[0] + op3.x + Σ c[d]*op0[d]; dst = outOff + Σ c[d]*op2[d]
// (op2 = destination strides — concat pieces on non-inner axes scatter)
const GATHER = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  let c = coords(g.x);
  var src = P.op3.x
          + c[0]*P.op0.x + c[1]*P.op0.y + c[2]*P.op0.z + c[3]*P.op0.w;
  let dst = P.outOff + c[0]*P.op2.x + c[1]*P.op2.y + c[2]*P.op2.z + c[3]*P.op2.w;
  A[dst] = ldInU(0u, src);
}
`;

// ---- flat copy (Reshape/Squeeze/Unsqueeze/Flatten with static shapes) ----
const COPY = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  A[P.outOff + g.x] = ldInU(0u, g.x);
}
`;

// ---- GatherElements: out[c] = in[c | axis = idx[c]] ----
// op0.x = axis; indices (int32) are input1 sharing the output shape
const GELEM = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  let c = coords(g.x);
  let iv = ldInU(1u, offOf(P.in1Stride, c));
  var ic = c;
  ic[P.op0.x] = iv;
  var src = 0u;
  for (var r: u32 = 0u; r < 4u; r++) { src += ic[r] * P.in0Stride[r]; }
  A[P.outOff + g.x] = ldInU(0u, src);
}
`;

// ---- reduce (Sum/Mean over ≤4 axes; membership-test walk) ----
// op0.x subop(0=sum,1=mean) | op0.y nIn | op0.w meanScale bits
// op4 = per-input-axis output-stride contribution (0 = reduced axis);
// handles keepdims 0/1 uniformly
const REDUCE = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  var acc = 0.0;
  for (var e: u32 = 0u; e < P.op0.y; e++) {
    var c = vec4<u32>(0u);
    var rem = e;
    for (var r: i32 = i32(P.rankOut) - 1; r >= 0; r--) {
      let d = P.in0Shape[u32(r)];
      c[u32(r)] = rem % d;
      rem = rem / d;
    }
    var oFlat = c[0]*P.op4.x + c[1]*P.op4.y + c[2]*P.op4.z + c[3]*P.op4.w;
    if (oFlat == g.x) { acc += ldIn(0u, e); }
  }
  if (P.op0.x == 1u) { acc = acc * bitcast<f32>(P.op0.w); }
  stfA(P.outOff + g.x, acc);
}
`;

// ---- softmax / logsoftmax along axis ----
// op0.x = axis, op0.y = logFlag; one workgroup per row
const SOFTMAX = PRELUDE + /* wgsl */`
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let rowLen = P.in0Shape[P.op0.x];
  let rowStride = P.in0Stride[P.op0.x];
  let nRows = P.nOut / rowLen;
  if (wid.x >= nRows) { return; }
  var inner = 1u;
  for (var r: u32 = P.op0.x + 1u; r < P.rankOut; r++) { inner *= P.outShape[r]; }
  let row = wid.x / inner;
  let sfx = wid.x % inner;
  let rowFlat = row * rowLen * inner + sfx;
  var m = -3.0e38;
  var i = lid.x;
  loop { if (i >= rowLen) { break; }
    m = max(m, ldfA(P.inOff.x + rowFlat + i * rowStride)); i += 256u; }
  red[lid.x] = m;
  workgroupBarrier();
  var s: u32 = 128u;
  loop { if (s == 0u) { break; }
    if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
    workgroupBarrier(); s = s / 2u; }
  let mx = red[0];
  var acc = 0.0;
  i = lid.x;
  loop { if (i >= rowLen) { break; }
    acc = acc + exp(ldfA(P.inOff.x + rowFlat + i * rowStride) - mx); i += 256u; }
  red[lid.x] = acc;
  workgroupBarrier();
  var s2: u32 = 128u;
  loop { if (s2 == 0u) { break; }
    if (lid.x < s2) { red[lid.x] = red[lid.x] + red[lid.x + s2]; }
    workgroupBarrier(); s2 = s2 / 2u; }
  let sum = red[0];
  let lse = mx + log(sum);
  i = lid.x;
  loop { if (i >= rowLen) { break; }
    let v = ldfA(P.inOff.x + rowFlat + i * rowStride);
    let o = select(exp(v - mx) / sum, v - lse, P.op0.y == 1u);
    stfA(P.outOff + rowFlat + i * rowStride, o); i += 256u; }
}
`;

// ---- batched fp32 matmul ----
// op0.x=M op0.y=N op0.z=K op0.w=batch | op1.x = bStride1 (0 if B unbatched)
const MATMUL_FP = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  let M = P.op0.x; let N = P.op0.y; let K = P.op0.z;
  let n = g.x % N;
  let m = (g.x / N) % M;
  let b = g.x / (N * M);
  var acc = 0.0;
  for (var k: u32 = 0u; k < K; k++) {
    let av = ldIn(0u, b * M * K + m * K + k);
    let bv = ldIn(1u, b * P.op1.x + k * N + n);
    acc = acc + av * bv;
  }
  stfA(P.outOff + g.x, acc);
}
`;

// ---- DQ stats: single workgroup computes min/scale/zp of input0 ----
// writes: A[outOff]=min, A[outOff+1]=scale, A[outOff+2]=zp(u32)
const QSTAT = PRELUDE + /* wgsl */`
var<workgroup> lo: array<f32, 64>;
var<workgroup> hi: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(global_invocation_id) g: vec3<u32>) {
  var mn = 3.0e38; var mx = -3.0e38;
  var i = lid.x;
  loop { if (i >= P.nOut) { break; }
    let v = ldIn(0u, i); mn = min(mn, v); mx = max(mx, v); i += 64u; }
  lo[lid.x] = mn; hi[lid.x] = mx;
  workgroupBarrier();
  if (lid.x == 0u) {
    var mn2 = 3.0e38; var mx2 = -3.0e38;
    for (var j: u32 = 0u; j < 64u; j++) {
      mn2 = min(mn2, lo[j]); mx2 = max(mx2, hi[j]);
    }
    var scale = (mx2 - mn2) / 255.0;
    if (scale == 0.0) { scale = 1.0; }
    // ONNX DQL (uint8): zp = qmin - round(qmin - min/scale), qmin = 0
    var zp = rhte(mn2 / scale);
    zp = clamp(zp, 0.0, 255.0);
    stfA(P.outOff, mn2);
    stfA(P.outOff + 1u, scale);
    A[P.outOff + 2u] = u32(zp);
    if (P.op2.y != 0u) { stfA(P.op2.y, scale); }   // DQL side: y_scale
    if (P.op2.z != 0u) { A[P.op2.z] = u32(zp); }   // DQL side: y_zero_point
  }
}
`;

// ---- quantized GEMM: out = aScale*wScale*Σ(q(a)-zp_a)*(w-zp_b), i32 exact ----
// op0.x=M op0.y=N op0.z=K | op1.x=wScale bits, op1.y=bZp(i32) | op2.x=scratch off
const QGEMM = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  let M = P.op0.x; let N = P.op0.y; let K = P.op0.z;
  let n = g.x % N;
  let m = (g.x / N) % M;
  let b = g.x / (N * M);
  let aScale = ldfA(P.op2.x + 1u);
  let aZp = i32(A[P.op2.x + 2u]);
  var wScale = bitcast<f32>(P.op1.x);
  if (P.op2.w != 0u) { wScale = wScale * ldfA(P.op2.w); }  // lazy-scale tensor
  let bZp = i32(P.op1.y);
  var acc = 0;
  for (var k: u32 = 0u; k < K; k++) {
    let av = ldfA(P.inOff.x + b * M * K + m * K + k);
    // quantize on the fly: q = clamp(round(a/scale)+zp, 0,255) - zp  (int32)
    let qr = clamp(i32(rhte(av / aScale)) + aZp, 0, 255) - aZp;
    let flat = k * N + n;
    let word = W[P.inOff.y + flat / 4u];
    let sh = (flat % 4u) * 8u;
    let wv = (i32(word << (24u - sh)) >> 24) - bZp;  // sign-extended int8
    acc = acc + qr * wv;
  }
  stfA(P.outOff + g.x, f32(acc) * aScale * wScale);
}
`;

// ---- 2D conv (direct loop; group covers depthwise) ----
// in [1,Cin,H,W] (batch 1), weight W-arena fp32 [Cout, CinG, KH, KW]
// op0.x=Cout op0.y=CinG op0.z=KH op0.w=KW
// op1.x=H_in op1.y=W_in op1.z=H_out op1.w=W_out
// op2.x=strideH op2.y=strideW op2.z=dilH op2.w=dilW
// op3.x=padTop op3.y=padLeft  (symmetric assumed; asymmetric -> fail at load)
const CONV = PRELUDE + /* wgsl */`
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.nOut) { return; }
  let Cout = P.op0.x; let CinG = P.op0.y; let KH = P.op0.z; let KW = P.op0.w;
  let H_in = P.op1.x; let W_in = P.op1.y; let H_out = P.op1.z; let W_out = P.op1.w;
  let group = P.in2Shape.x;  // packed: group count in in2Shape.x slot
  let ow = g.x % W_out;
  let oh = (g.x / W_out) % H_out;
  let co = g.x / (W_out * H_out);
  var acc = 0.0;
  // grouped conv: input channel base for co's group (group=1 -> always 0)
  let cin0 = (co / (Cout / group)) * CinG;
  for (var ci: u32 = 0u; ci < CinG; ci++) {
    for (var kh: u32 = 0u; kh < KH; kh++) {
      let ih = oh * P.op2.x + kh * P.op2.z - P.op3.x;
      if (ih >= H_in) { continue; }
      for (var kw: u32 = 0u; kw < KW; kw++) {
        let iw = ow * P.op2.y + kw * P.op2.w - P.op3.y;
        if (iw >= W_in) { continue; }
        let aOff = (cin0 + ci) * H_in * W_in + ih * W_in + iw;
        let wOff = ((co * CinG + ci) * KH + kh) * KW + kw;
        acc = acc + ldIn(0u, aOff) * ldIn(1u, wOff);
      }
    }
  }
  if (P.inCount > 2u) { acc = acc + ldIn(2u, co); }  // bias [Cout]
  stfA(P.outOff + g.x, acc);
}
`;

export const PIPE_SOURCES = {
  elem: ELEM, gather: GATHER, copy: COPY, gelem: GELEM, reduce: REDUCE,
  softmax: SOFTMAX, matmul_fp: MATMUL_FP, qstat: QSTAT, qgemm: QGEMM,
  conv: CONV,
};
