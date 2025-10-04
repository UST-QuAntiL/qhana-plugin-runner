caches.keys().then(keys => keys.forEach(key => caches.delete(key)))
