// 站点根 Service Worker —— a1 实验专用（其他页面不注册则完全不受影响）
// 策略：对 4 个大资产（wasm js/wasm/data + glue）cache-first；首次访问 install 预热，二次 0 字节。
const CACHE = 'speedlab-v1';
const BASE = '/asr-wasm-speedlab';
const ASSETS = [
  BASE + '/sherpa-onnx-asr.js',
  BASE + '/sherpa-onnx-wasm-main-asr.js',
  BASE + '/sherpa-onnx-wasm-main-asr.wasm',
  BASE + '/sherpa-onnx-wasm-main-asr.data',
];

self.addEventListener('install', e => {
  e.waitUntil(
    caches.open(CACHE)
      .then(c => c.addAll(ASSETS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys()
      .then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim()) // 立即接管当前页，首次访问即可吃到 install 预热
  );
});

self.addEventListener('fetch', e => {
  const u = new URL(e.request.url);
  if (!ASSETS.includes(u.pathname)) return; // 其他请求一律直通
  e.respondWith(
    caches.open(CACHE).then(c =>
      c.match(e.request).then(hit => {
        if (hit) return hit;
        return fetch(e.request).then(resp => {
          if (resp.ok) c.put(e.request, resp.clone());
          return resp;
        });
      })
    )
  );
});
