{
    function clearCache() {
	caches.keys().then(keys => keys.forEach(key => caches.delete(key)))
    }
    setTimeout(clearCache, 1000)
    setTimeout(clearCache)
}
