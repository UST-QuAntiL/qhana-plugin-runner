"use strict";
{
    /**
     * Apply various workarounds
     */
    function applyWorkarounds() {
	// clear workbox cache
	window.caches.keys()
	    .then(keys => Promise.all(keys.map(key => caches.delete(key))))
	    .then(() => console.log("cache has been cleared!"))

	// request full height
	if (window._qhana_microfrontend_state) {
	    window._qhana_microfrontend_state.targetHeight = "full"
	    window.notifyParentWindowOnHeightChange(null, "full")
	    console.log("full height requested!")
	} else {
	    console.log("microfrontend.js hasn't been loaded yet, cannot request full height!")
	}

	// ignore css load requests
	window.onLoadCssMessage = function onLoadCssMessage(data, state) {
	    console.log("css load request from the ui has been ignored!", data, state)
	}
    }
    // apply workarounds in 1000ms
    setTimeout(applyWorkarounds, 1000)
    // apply workarounds in the next event cycle
    setTimeout(applyWorkarounds)
    // apply workarounds now
    applyWorkarounds()
}
