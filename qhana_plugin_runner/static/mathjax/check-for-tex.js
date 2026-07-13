(function () {
    const body = document.body?.textContent ?? '';
    if (!body.match(/(?:\$|\\\(|\\\[|\\begin\{.*?})/)) {
        return;
    }

    if (!window.MathJax) {
        window.MathJax = {
            tex: {
                inlineMath: { '[+]': [['$', '$']] }
            }
        };
    }


    const config = window.qhanaMathJax || {};

    const script = document.createElement('script');
    script.src = config.src || '/static/mathjax/es5/tex-mml-chtml.js';
    if (config.integrity) {
        script.integrity = config.integrity;
    }
    script.crossOrigin = "anonymous";
    script.async = true;
    document.head.appendChild(script);
})();