const CACHE = 'yf-tracker-review-shell-v2';
const ASSETS = ['./', 'index.html', 'review-store.js', 'manifest.json', 'icon-192.png'];
self.addEventListener('install', event => {
    event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(ASSETS)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', event => {
    event.waitUntil(caches.keys().then(names => Promise.all(names.filter(name => name.startsWith('yf-tracker-review-shell-') && name !== CACHE).map(name => caches.delete(name)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);
    if (event.request.method !== 'GET' || url.origin !== self.location.origin || !url.href.startsWith(self.registration.scope)) return;
    event.respondWith(fetch(event.request).then(response => {
        if (response.ok) { const copy = response.clone(); event.waitUntil(caches.open(CACHE).then(cache => cache.put(event.request, copy))); }
        return response;
    }).catch(() => caches.match(event.request).then(response => response || (event.request.mode === 'navigate' ? caches.match('index.html') : Response.error()))));
});