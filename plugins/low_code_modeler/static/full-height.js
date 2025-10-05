{
    function forceFullHeight() {
	window._qhana_microfrontend_state.targetHeight = "full"
	notifyParentWindowOnHeightChange(null, "full")
    }
    setTimeout(forceFullHeight, 1000)
    setTimeout(forceFullHeight)
}
