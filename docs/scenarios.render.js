(function(){
  var grid = document.getElementById('scenarioGrid');
  if (!grid || typeof SCENARIOS === 'undefined') return;

  function bounds(s){
    var all = s.history_xy.concat(s.ground_truth_xy, s.predicted_xy);
    var lateral = all.map(function(p){ return -p[1]; });
    var forward = all.map(function(p){ return p[0]; });
    var minX = Math.min.apply(null, lateral), maxX = Math.max.apply(null, lateral);
    var minY = Math.min.apply(null, forward), maxY = Math.max.apply(null, forward);
    var mx = (maxX - minX) * 0.15 + 0.6, my = (maxY - minY) * 0.08 + 0.6;
    return [minX - mx, maxX + mx, minY - my, maxY + my];
  }

  function drawScenario(canvas, s, animate, onDone){
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    var ctx = canvas.getContext('2d');
    var b = bounds(s), pad = 10 * dpr;
    function toPx(p){
      var x = -p[1], y = p[0];
      return [
        pad + (x - b[0]) / (b[1] - b[0]) * (canvas.width - 2 * pad),
        canvas.height - pad - (y - b[2]) / (b[3] - b[2]) * (canvas.height - 2 * pad)
      ];
    }
    function drawLine(pts, color, width){
      if (pts.length < 2) return;
      ctx.beginPath();
      pts.forEach(function(p, i){
        var px = toPx(p);
        if (i === 0) ctx.moveTo(px[0], px[1]); else ctx.lineTo(px[0], px[1]);
      });
      ctx.strokeStyle = color; ctx.lineWidth = width * dpr;
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.stroke();
    }
    function render(frac){
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      drawLine(s.history_xy, '#51565f', 2);
      var gtN = Math.max(2, Math.round(s.ground_truth_xy.length * frac));
      var prN = Math.max(2, Math.round(s.predicted_xy.length * frac));
      drawLine(s.ground_truth_xy.slice(0, gtN), '#F2718A', 2.2);
      drawLine(s.predicted_xy.slice(0, prN), '#4FA8FF', 2.6);
    }
    if (!animate) { render(1); return; }
    var start = null, duration = 1200;
    function step(ts){
      if (start === null) start = ts;
      var frac = Math.min((ts - start) / duration, 1);
      render(frac);
      if (frac < 1) requestAnimationFrame(step);
      else if (onDone) onDone();
    }
    requestAnimationFrame(step);
  }

  function loopScenario(canvas, s){
    drawScenario(canvas, s, true, function(){
      setTimeout(function(){ loopScenario(canvas, s); }, 900);
    });
  }

  SCENARIOS.forEach(function(s){
    var card = document.createElement('div');
    card.className = 'scenario-card';
    var canvas = document.createElement('canvas');
    card.appendChild(canvas);
    var meta = document.createElement('div');
    meta.className = 'meta';
    meta.innerHTML = '<span class="cid">' + s.clip_id.slice(0, 8) + '&hellip;</span><span class="ade">minADE ' + s.min_ade_m.toFixed(3) + 'm</span>';
    card.appendChild(meta);
    var reason = document.createElement('div');
    reason.className = 'reason';
    reason.textContent = s.reasoning;
    card.appendChild(reason);
    grid.appendChild(card);
    requestAnimationFrame(function(){ loopScenario(canvas, s); });
    card.addEventListener('click', function(){ drawScenario(canvas, s, true); });
  });
})();
