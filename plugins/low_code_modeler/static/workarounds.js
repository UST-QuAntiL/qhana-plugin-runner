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

    // apply workarounds the first time a message is recived
    // because this event listener is registered before microfrontend.js is executed, this one should be executed first

    let isFirstMessage = true
    window.addEventListener("message", (event) => {
	if (!isFirstMessage)
	    return
	isFirstMessage = false
	applyWorkarounds()
    })
}
