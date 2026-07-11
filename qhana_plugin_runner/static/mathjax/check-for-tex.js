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

    const script = document.createElement('script');
    script.src = '/static/mathjax/es5/tex-mml-chtml.js';
    script.integrity = "sha384-Wuix6BuhrWbjDBs24bXrjf4ZQ5aFeFWBuKkFekO2t8xFU0iNaLQfp2K6/1Nxveei";
    script.crossOrigin = "anonymous";
    script.async = true;
    document.head.appendChild(script);
})();