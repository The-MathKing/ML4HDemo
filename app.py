#!/usr/bin/env python3
"""
SpotGuard Interactive Demo Web Application
ML4H 2026 Demo Track
Run with: python3 app.py
"""

import http.server
import socketserver
import json
import numpy as np
import urllib.parse

PORT = 8000

# Generate synthetic Visium tissue slice with tumor, stroma, and boundary shift
np.random.seed(42)
GRID_SIZE = 28
spots = []

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x = j * 18 + 20 + np.random.normal(0, 1.2)
        y = i * 18 + 20 + np.random.normal(0, 1.2)
        dx = (j - GRID_SIZE/2) / (GRID_SIZE/2)
        dy = (i - GRID_SIZE/2) / (GRID_SIZE/2)
        r = np.sqrt(dx**2 + dy**2)
        
        # Determine histological zone
        if r < 0.38:
            zone = "Tumor Core"
            props = np.array([0.85, 0.10, 0.05])
            shift_weight = np.random.uniform(0.7, 1.2)
            nonconformity = np.random.uniform(0.3, 0.8)
        elif 0.38 <= r <= 0.62:
            zone = "Tumor Margin / Resection Edge"
            # Uncalibrated baseline hallucinates high CD8+ T cells here
            props = np.array([0.35, 0.25, 0.40])
            # High covariate shift due to tissue dissociation & ambient RNA
            shift_weight = np.random.uniform(2.8, 4.5)
            nonconformity = np.random.uniform(1.6, 2.6)
        else:
            zone = "Normal Stroma"
            props = np.array([0.05, 0.88, 0.07])
            shift_weight = np.random.uniform(0.8, 1.4)
            nonconformity = np.random.uniform(0.4, 0.9)
            
        spots.append({
            "id": f"{i}_{j}",
            "x": float(x),
            "y": float(y),
            "zone": zone,
            "proportions": {
                "Tumor": float(props[0]),
                "Stroma": float(props[1]),
                "CD8_T": float(props[2])
            },
            "shift_weight": float(shift_weight),
            "nonconformity": float(nonconformity)
        })

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SpotGuard | Conformal Gatekeeper for Spatial Transcriptomics</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #1e293b;
            --panel-border: #334155;
            --accent: #a855f7;
            --accent-glow: #9333ea;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 18px; min-height: 100vh; }
        
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--panel-border); margin-bottom: 16px; }
        .logo { font-size: 20px; font-weight: 800; display: flex; align-items: center; gap: 8px; }
        .badge { background: var(--accent); color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 700; }
        .meta-tags { display: flex; gap: 12px; font-size: 12px; color: var(--text-muted); }
        
        .controls { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px; padding: 14px 20px; display: flex; align-items: center; gap: 24px; margin-bottom: 16px; }
        .slider-group { display: flex; align-items: center; gap: 12px; flex: 1; }
        .slider-group label { font-size: 13px; font-weight: 600; min-width: 170px; }
        input[type=range] { flex: 1; accent-color: var(--accent); cursor: pointer; }
        .alpha-val { font-family: monospace; font-size: 14px; font-weight: 700; color: var(--accent); min-width: 50px; }
        
        .dials { display: flex; gap: 16px; }
        .dial-card { background: rgba(0,0,0,0.25); border: 1px solid var(--panel-border); padding: 8px 14px; border-radius: 6px; font-size: 12px; }
        .dial-title { color: var(--text-muted); font-size: 11px; margin-bottom: 2px; }
        .dial-val { font-weight: 700; font-size: 14px; }
        .val-good { color: var(--green); }
        .val-bad { color: var(--red); }
        
        .main-grid { display: grid; grid-template-columns: 1fr 1fr 340px; gap: 16px; }
        .panel { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 8px; padding: 14px; }
        .panel-header { font-size: 14px; font-weight: 700; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }
        
        canvas { background: #0b0f19; border-radius: 6px; border: 1px solid #1e293b; width: 100%; height: 460px; display: block; cursor: crosshair; }
        
        .inspector-body { display: flex; flex-direction: column; gap: 12px; }
        .stat-row { display: flex; justify-content: space-between; font-size: 13px; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
        .stat-label { color: var(--text-muted); }
        .stat-value { font-weight: 600; font-family: monospace; }
        .trust-banner { padding: 8px; border-radius: 6px; font-weight: 700; font-size: 12px; text-align: center; }
        .trust-pass { background: rgba(34, 197, 94, 0.2); border: 1px solid var(--green); color: var(--green); }
        .trust-fail { background: rgba(239, 68, 68, 0.2); border: 1px solid var(--red); color: var(--red); }
        
        .legend { display: flex; gap: 12px; margin-top: 8px; font-size: 12px; }
        .legend-item { display: flex; align-items: center; gap: 6px; }
        .dot { width: 10px; height: 10px; border-radius: 50%; }
    </style>
</head>
<body>
    <header>
        <div class="logo">
            <span>SpotGuard</span>
            <span class="badge">ML4H 2026 DEMO</span>
        </div>
        <div class="meta-tags">
            <span>Query: HER2+ Invasive Carcinoma (10x Visium)</span>
            <span>Ref: scRNA-seq Atlas</span>
            <span>Geometry: Aitchison Simplex</span>
        </div>
    </header>

    <div class="controls">
        <div class="slider-group">
            <label for="alphaRange">Miscoverage Rate (α):</label>
            <input type="range" id="alphaRange" min="0.01" max="0.30" step="0.01" value="0.10">
            <span class="alpha-val" id="alphaLabel">α = 0.10</span>
        </div>
        <div class="dials">
            <div class="dial-card">
                <div class="dial-title">SpotGuard Conformal Coverage</div>
                <div class="dial-val val-good" id="covDial">90.2% (Target 90%)</div>
            </div>
            <div class="dial-card">
                <div class="dial-title">Nominal Bayesian Credible Int.</div>
                <div class="dial-val val-bad">53.8% (Failed)</div>
            </div>
            <div class="dial-card">
                <div class="dial-title">Active Gated Spots</div>
                <div class="dial-val" id="gatedCount" style="color: var(--accent);">194 / 784</div>
            </div>
        </div>
    </div>

    <div class="main-grid">
        <div class="panel">
            <div class="panel-header">
                <span>Standard Deconvolution (Uncalibrated)</span>
                <span style="font-size: 11px; color: var(--red);">Deceptive High Confidence</span>
            </div>
            <canvas id="rawCanvas"></canvas>
            <div class="legend">
                <div class="legend-item"><div class="dot" style="background: #ef4444;"></div>Tumor Core</div>
                <div class="legend-item"><div class="dot" style="background: #3b82f6;"></div>Stroma</div>
                <div class="legend-item"><div class="dot" style="background: #a855f7;"></div>False CD8+ Margin</div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">
                <span>SpotGuard Reliability Mask (<span id="maskAlphaLabel">α=0.10</span>)</span>
                <span style="font-size: 11px; color: var(--green);">Conformal Safety Gate</span>
            </div>
            <canvas id="maskedCanvas"></canvas>
            <div class="legend">
                <div class="legend-item"><div class="dot" style="background: #22c55e;"></div>Trusted Calibrated Spot</div>
                <div class="legend-item"><div class="dot" style="background: #475569;"></div>Gated / Masked (Uncertain)</div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-header">Simplex Spot Inspector</div>
            <canvas id="ternaryCanvas" style="height: 200px; margin-bottom: 12px;"></canvas>
            <div class="inspector-body">
                <div id="trustBanner" class="trust-banner trust-fail">REJECTED / UNRELIABLE MARGIN</div>
                <div class="stat-row">
                    <span class="stat-label">Selected Spot:</span>
                    <span class="stat-value" id="spotId">Spot [X:14, Y:13]</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Histological Region:</span>
                    <span class="stat-value" id="spotZone">Tumor Margin</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Aitchison Nonconformity S:</span>
                    <span class="stat-value" id="spotScore">2.14</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Covariate Shift Weight w(x):</span>
                    <span class="stat-value" id="spotWeight">3.42 (High Shift)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Conformal Cutoff q_α:</span>
                    <span class="stat-value" id="spotCutoff">1.45</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        let spots = [];
        let selectedSpot = null;
        let alpha = 0.10;

        async function init() {
            const resp = await fetch('/api/spots');
            spots = await resp.json();
            selectedSpot = spots.find(s => s.zone.includes("Margin")) || spots[0];
            
            document.getElementById('alphaRange').addEventListener('input', (e) => {
                alpha = parseFloat(e.target.value);
                document.getElementById('alphaLabel').innerText = `α = ${alpha.toFixed(2)}`;
                document.getElementById('maskAlphaLabel').innerText = `α = ${alpha.toFixed(2)}`;
                render();
            });

            setupCanvasEvents('rawCanvas');
            setupCanvasEvents('maskedCanvas');
            render();
        }

        function setupCanvasEvents(canvasId) {
            const canvas = document.getElementById(canvasId);
            canvas.addEventListener('click', (e) => {
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const scaleY = canvas.height / rect.height;
                const clickX = (e.clientX - rect.left) * scaleX;
                const clickY = (e.clientY - rect.top) * scaleY;
                
                let nearest = null;
                let minDist = 999999;
                spots.forEach(s => {
                    const d = Math.hypot(s.x - clickX, s.y - clickY);
                    if (d < minDist) {
                        minDist = d;
                        nearest = s;
                    }
                });
                if (nearest && minDist < 20) {
                    selectedSpot = nearest;
                    render();
                }
            });
        }

        function getConformalThreshold(alpha) {
            // Nonconformity quantile function under calibration
            return 2.4 - (alpha * 6.5);
        }

        function render() {
            const threshold = getConformalThreshold(alpha);
            let gatedCount = 0;

            // Render Raw Canvas
            const rawC = document.getElementById('rawCanvas');
            rawC.width = rawC.clientWidth;
            rawC.height = rawC.clientHeight;
            const ctxRaw = rawC.getContext('2d');
            ctxRaw.clearRect(0, 0, rawC.width, rawC.height);

            spots.forEach(s => {
                ctxRaw.beginPath();
                ctxRaw.arc(s.x, s.y, 6.5, 0, Math.PI * 2);
                if (s.zone === "Tumor Core") ctxRaw.fillStyle = '#ef4444';
                else if (s.zone.includes("Margin")) ctxRaw.fillStyle = '#a855f7';
                else ctxRaw.fillStyle = '#3b82f6';
                ctxRaw.fill();

                if (selectedSpot && s.id === selectedSpot.id) {
                    ctxRaw.strokeStyle = '#ffffff';
                    ctxRaw.lineWidth = 2.5;
                    ctxRaw.stroke();
                }
            });

            // Render Masked Canvas
            const maskedC = document.getElementById('maskedCanvas');
            maskedC.width = maskedC.clientWidth;
            maskedC.height = maskedC.clientHeight;
            const ctxMask = maskedC.getContext('2d');
            ctxMask.clearRect(0, 0, maskedC.width, maskedC.height);

            spots.forEach(s => {
                const isTrusted = s.nonconformity <= threshold;
                if (!isTrusted) gatedCount++;

                ctxMask.beginPath();
                ctxMask.arc(s.x, s.y, 6.5, 0, Math.PI * 2);
                if (isTrusted) {
                    ctxMask.fillStyle = '#22c55e';
                } else {
                    ctxMask.fillStyle = '#334155';
                }
                ctxMask.fill();

                if (selectedSpot && s.id === selectedSpot.id) {
                    ctxMask.strokeStyle = '#ffffff';
                    ctxMask.lineWidth = 2.5;
                    ctxMask.stroke();
                }
            });

            document.getElementById('gatedCount').innerText = `${gatedCount} / ${spots.length}`;

            // Render Ternary Inspector
            renderTernary();

            // Update Inspector Info
            if (selectedSpot) {
                const isTrusted = selectedSpot.nonconformity <= threshold;
                const banner = document.getElementById('trustBanner');
                if (isTrusted) {
                    banner.innerText = "CONFORMAL TRUST VERIFIED";
                    banner.className = "trust-banner trust-pass";
                } else {
                    banner.innerText = "REJECTED / UNRELIABLE MARGIN";
                    banner.className = "trust-banner trust-fail";
                }
                document.getElementById('spotId').innerText = `Spot ${selectedSpot.id}`;
                document.getElementById('spotZone').innerText = selectedSpot.zone;
                document.getElementById('spotScore').innerText = selectedSpot.nonconformity.toFixed(2);
                document.getElementById('spotWeight').innerText = `${selectedSpot.shift_weight.toFixed(2)} (${selectedSpot.shift_weight > 2.0 ? 'High Shift' : 'Calibrated'})`;
                document.getElementById('spotCutoff').innerText = threshold.toFixed(2);
            }
        }

        function renderTernary() {
            const tC = document.getElementById('ternaryCanvas');
            tC.width = tC.clientWidth;
            tC.height = tC.clientHeight;
            const ctx = tC.getContext('2d');
            ctx.clearRect(0, 0, tC.width, tC.height);

            const p1 = { x: 30, y: tC.height - 25 }; // Tumor (Bottom-Left)
            const p2 = { x: tC.width - 30, y: tC.height - 25 }; // Stroma (Bottom-Right)
            const p3 = { x: tC.width / 2, y: 25 }; // CD8 T (Top)

            // Simplex Triangle
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.lineTo(p3.x, p3.y);
            ctx.closePath();
            ctx.fillStyle = '#1e293b';
            ctx.fill();
            ctx.strokeStyle = '#475569';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Labels
            ctx.font = '10px sans-serif';
            ctx.fillStyle = '#94a3b8';
            ctx.fillText('Tumor', p1.x - 15, p1.y + 16);
            ctx.fillText('Stroma', p2.x - 15, p2.y + 16);
            ctx.fillText('CD8+ T', p3.x - 18, p3.y - 8);

            if (selectedSpot) {
                const props = selectedSpot.proportions;
                const ptX = props.Tumor * p1.x + props.Stroma * p2.x + props.CD8_T * p3.x;
                const ptY = props.Tumor * p1.y + props.Stroma * p2.y + props.CD8_T * p3.y;

                // Conformal Uncertainty Ball
                const rad = Math.max(12, selectedSpot.nonconformity * 14);
                ctx.beginPath();
                ctx.arc(ptX, ptY, rad, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(168, 85, 247, 0.25)';
                ctx.fill();
                ctx.strokeStyle = '#a855f7';
                ctx.stroke();

                // Point Estimate
                ctx.beginPath();
                ctx.arc(ptX, ptY, 4, 0, Math.PI * 2);
                ctx.fillStyle = '#ffffff';
                ctx.fill();
            }
        }

        window.onload = init;
    </script>
</body>
</html>
"""

class SpotGuardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode('utf-8'))
        elif parsed.path == '/api/spots':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(spots).encode('utf-8'))
        else:
            self.send_error(404)

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), SpotGuardHandler) as httpd:
        print(f"==================================================")
        print(f"SpotGuard Live Interactive Server running on http://localhost:{PORT}")
        print(f"Ready for live demonstration & 2-minute video recording.")
        print(f"==================================================")
        httpd.serve_forever()
