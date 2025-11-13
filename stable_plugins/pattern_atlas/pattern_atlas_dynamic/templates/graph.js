window.PatternGraph = {

enableZoom(svgId = "graph", groupSelector = ".zoom-group", markerId = "arrow", kMin = 0.2, kMax = 8) {
    const svg = document.getElementById(svgId);
    const marker = document.getElementById(markerId);
    const viewport = svg.querySelector(groupSelector);
    let k = 1, x = 0, y = 0;

    const apply = () => {
        viewport.setAttribute("transform", `translate(${x},${y}) scale(${k})`);
        marker.setAttribute("markerWidth", `${k * 8}`);
        marker.setAttribute("markerHeight", `${k * 8}`);
    }
    apply();

    const svgPoint = (evt) => {
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX; 
        pt.y = evt.clientY;
        return pt.matrixTransform(svg.getScreenCTM().inverse());
    };

    const onWheel = (e) => {
        e.preventDefault();
        const f = e.deltaY < 0 ? 1.1 : 0.9;
        const nk = Math.min(kMax, Math.max(kMin, k * f));
        const c = svgPoint(e);
        x = c.x - (c.x - x) * (nk / k);
        y = c.y - (c.y - y) * (nk / k);
        k = nk;
        apply();
    };

    svg.addEventListener("wheel", onWheel, { passive: false, capture: true });
},

layoutRandom(svgId = "graph", nodeSel = ".node", margin = 40) {
    const svg = document.getElementById(svgId);
    const nodes = svg.querySelectorAll(nodeSel);
    const vb = svg.viewBox.baseVal;
    const W = vb.width, H = vb.height;

    nodes.forEach(node => {
        const x = margin + Math.random() * (W - 2*margin);
        const y = margin + Math.random() * (H - 2*margin);
        node.setAttribute("transform", `translate(${x.toFixed(1)}, ${y.toFixed(1)})`);
        node.setAttribute("data-x", x.toFixed(1));
        node.setAttribute("data-y", y.toFixed(1));
    });
},

connectEdges(svgId = "graph", edgeSel = ".edge", nodeSel = ".node") {
    const svg = document.getElementById(svgId);
    const edges = svg.querySelectorAll(edgeSel);
    const nodes = svg.querySelectorAll(nodeSel);
    let x1 = 0;
    let y1 = 0;
    let x2 = 0;
    let y2 = 0;

    edges.forEach(edge => {
        const source = edge.dataset.source;
        const target = edge.dataset.target;
        for (const node of nodes) {
            const x = node.dataset.x;
            const y = node.dataset.y;
            let startSet = false;
            let endSet = false;
            if (node.getAttribute("id") === "node-" + source) {
                x1 = parseFloat(x);
                y1 = parseFloat(y);
                startSet = true;
            }
            else if (node.getAttribute("id") === "node-" + target) {
                x2 = parseFloat(x);
                y2 = parseFloat(y);
                endSet = true;
            }
            if (startSet && endSet) {
                break;
            }
        }
        if (y1 > y2) {
            y1 = y1 - 25;
            y2 = y2 + 25;
        }
        else {
            y1 = y1 + 25;
            y2 = y2 - 25;
        }
        edge.setAttribute("d", this.pathDCubic(x1, y1, x2, y2, 0.5));
    })
},

pathDCubic(x1, y1, x2, y2, t = 0.5) {
    const dx = this.distance(x1, x2), dy = this.distance(y1, y2);
    const cx1 = x1 + dx * t;
    const cy1 = y1 + dy * t;
    const cx2 = x2 - dx * t;
    const cy2 = y2 - dy * t;
    return `M ${x1} ${y1} C ${cx1} ${cy1}, ${cx2} ${cy2}, ${x2} ${y2}`;
},

distance(x, y) {
    if (x > y) {
        return x - y;
    }
    return y - x;
}

};
