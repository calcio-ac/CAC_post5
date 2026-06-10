"""
CAC_post5 — France WC 2022 passing-network film.

Renders a 16:9 4K video (77 s, 24 fps): one passing network per France match,
morphing between games, with possession / passes / xG callouts, and an
all-seven finale. Synth ambient score generated in numpy, muxed with ffmpeg.

  python3 make_video.py            # full 4K render
  python3 make_video.py --preview  # 1280x720, every 3rd frame, no audio
"""
import json, math, os, subprocess, sys, wave

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, Rectangle
import imageio_ffmpeg

PREVIEW = '--preview' in sys.argv

# ---------------------------------------------------------------- palette
NAVY, BLUE, RED, WHITE, TINT = '#21304D', '#17548C', '#ED2939', '#FFFFFF', '#9DBBDD'
DARK = '#161F33'
FONT = 'Courier New'

FPS = 24
W, H = 1920, 1080                    # design space; rendered at 2x for 4K
SCALE = 1.0 if PREVIEW else 2.0
STEP = 3 if PREVIEW else 1

TITLE_S, MATCH_S, TRANS_S, FINALE_S = 5.0, 7.0, 2.0, 9.0

# ---------------------------------------------------------------- data
DATA = json.load(open('web/data.json'))['matches']
N = len(DATA)

# per-match stats computed from StatsBomb events (see notebook / analysis)
STATS = {
    'Australia': dict(poss=61.6, xg='3.17 – 0.32'),
    'Denmark':   dict(poss=48.8, xg='1.93 – 0.89'),
    'Tunisia':   dict(poss=64.7, xg='0.48 – 0.30'),
    'Poland':    dict(poss=54.9, xg='1.13 – 1.59'),
    'England':   dict(poss=42.7, xg='0.84 – 2.35'),
    'Morocco':   dict(poss=38.6, xg='2.00 – 1.23'),
    'Argentina': dict(poss=45.1, xg='2.27 – 2.76'),
}
CALLOUTS = {
    'Australia': ['Lucas Hernández lasts 12 minutes.',
                  'Theo inherits the flank — the shape never blinks.'],
    'Denmark':   ['Tchouaméni becomes the hub.',
                  'The spine tilts left through Theo and Rabiot.'],
    'Tunisia':   ['Nine changes, same skeleton.',
                  'The B-side keeps the ball, creates 0.48 xG.'],
    'Poland':    ['The right side wakes up.',
                  'Koundé to Dembélé becomes the release valve.'],
    'England':   ['Out-passed. Out-xG\'d. Through.',
                  'Everything funnels down the right — France win anyway.'],
    'Morocco':   ['38% of the ball, by design.',
                  'Varane is the hub now: defend, then detonate.'],
    'Argentina': ['Pinned for 80 minutes, the network sinks.',
                  'Then Mbappé drags the final back from the dead.'],
}

# ---------------------------------------------------------------- layout
PX, PY, PW, PH = 70, 200, 1010, 673          # pitch box (120x80 SB, attack ->)
PANEL_X = 1170

def sx(x): return PX + (x / 120) * PW
def sy(y): return PY + (y / 80) * PH

# collision-relaxed node positions per match, in screen units
def relax(nodes):
    maxc = max(n['count'] for n in nodes)
    pts = {n['name']: dict(x=sx(n['x']), y=sy(n['y']),
                           r=20 + 17 * (n['count'] / maxc), n=n) for n in nodes}
    arr = list(pts.values())
    for _ in range(120):
        moved = False
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                a, b = arr[i], arr[j]
                dx, dy = b['x'] - a['x'], b['y'] - a['y']
                d = max(math.hypot(dx, dy), 1e-6)
                mind = a['r'] + b['r'] + 26          # room for labels
                if d < mind:
                    push = (mind - d) / 2
                    dx, dy = dx / d, dy / d
                    a['x'] -= dx * push; a['y'] -= dy * push
                    b['x'] += dx * push; b['y'] += dy * push
                    moved = True
        for p in arr:
            p['x'] = min(max(p['x'], PX + p['r'] + 6), PX + PW - p['r'] - 6)
            p['y'] = min(max(p['y'], PY + p['r'] + 6), PY + PH - p['r'] - 22)
        if not moved:
            break
    return pts

LAYOUTS = [relax(m['nodes']) for m in DATA]
MAXN = [max(e['n'] for e in m['edges']) for m in DATA]

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(19.2, 10.8), dpi=100 * SCALE)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
ax.set_xlim(0, W); ax.set_ylim(H, 0)
fig.patch.set_facecolor(NAVY)

rng = np.random.default_rng(7)
PARTICLES = rng.uniform([0, 0, .04, .2], [W, H, .12, 1.0], (46, 4))
PVEL = rng.uniform([-7, -4], [7, 4], (46, 2))

def ease(t):  # smootherstep
    t = min(max(t, 0), 1)
    return t * t * t * (t * (6 * t - 15) + 10)

def txt(x, y, s, size, color=WHITE, weight='bold', ha='left', alpha=1.0, va='top',
        zorder=4):
    if alpha <= 0.01:
        return []
    return [ax.text(x, y, s, fontsize=size * 0.72, family=FONT, fontweight=weight,
                    color=color, ha=ha, va=va, alpha=alpha, zorder=zorder)]

def draw_particles(t, alpha=1.0):
    p = PARTICLES.copy()
    p[:, 0] = (p[:, 0] + PVEL[:, 0] * t) % W
    p[:, 1] = (p[:, 1] + PVEL[:, 1] * t) % H
    return [ax.scatter(p[:, 0], p[:, 1], s=p[:, 3] * 14 * SCALE, color=TINT,
                       alpha=0.10 * alpha, linewidths=0)]

def draw_pitch(alpha=1.0, px=PX, py=PY, pw=PW, ph=PH, lw=2.0):
    A, arts = alpha * 0.75, []
    def L(x0, y0, x1, y1):
        arts.append(ax.plot([x0, x1], [y0, y1], color=WHITE, lw=lw * SCALE, alpha=A,
                            solid_capstyle='round')[0])
    X = lambda v: px + v / 120 * pw
    Y = lambda v: py + v / 80 * ph
    for a, b, c, d in [(0, 0, 120, 0), (120, 0, 120, 80), (120, 80, 0, 80), (0, 80, 0, 0),
                       (60, 0, 60, 80),
                       (0, 18, 18, 18), (18, 18, 18, 62), (0, 62, 18, 62),
                       (120, 18, 102, 18), (102, 18, 102, 62), (120, 62, 102, 62),
                       (0, 30, 6, 30), (6, 30, 6, 50), (0, 50, 6, 50),
                       (120, 30, 114, 30), (114, 30, 114, 50), (120, 50, 114, 50)]:
        L(X(a), Y(b), X(c), Y(d))
    r = 10 / 120 * pw
    arts.append(ax.add_patch(Circle((X(60), Y(40)), r, fill=False, ec=WHITE,
                                    lw=lw * SCALE, alpha=A)))
    for spot, th in [(X(12), (-53, 53)), (X(108), (127, 233))]:
        arts.append(ax.add_patch(Arc((spot, Y(40)), 2 * r, 2 * r * (ph / pw) * (120 / 80),
                                     theta1=th[0], theta2=th[1], ec=WHITE,
                                     lw=lw * SCALE, alpha=A)))
    return arts

def lerp_layout(i, j, t):
    """Blend node positions/sizes between match i and j; alpha handles in/out."""
    t = ease(t)
    out = {}
    A, B = LAYOUTS[i], LAYOUTS[j]
    for name in set(A) | set(B):
        a, b = A.get(name), B.get(name)
        if a and b:
            out[name] = dict(x=a['x'] + (b['x'] - a['x']) * t,
                             y=a['y'] + (b['y'] - a['y']) * t,
                             r=a['r'] + (b['r'] - a['r']) * t,
                             alpha=1.0, n=b['n'])
        elif a:
            out[name] = dict(x=a['x'], y=a['y'], r=a['r'] * (1 - 0.3 * t),
                             alpha=1 - t, n=a['n'])
        else:
            out[name] = dict(x=b['x'], y=b['y'], r=b['r'] * (0.7 + 0.3 * t),
                             alpha=t, n=b['n'])
    return out

def draw_network(mi, pts, alpha=1.0, growth=1.0, pulse=0.0, edge_blend=None):
    """edge_blend = (other_match_index, t) to crossfade edge weights."""
    arts = []
    m = DATA[mi]
    edges = {(e['a'], e['b']): e['n'] for e in m['edges'] if e['n'] >= 2}
    maxn = MAXN[mi]
    if edge_blend:
        oi, t = edge_blend
        other = {(e['a'], e['b']): e['n'] for e in DATA[oi]['edges'] if e['n'] >= 2}
        keys = set(edges) | set(other)
        edges = {k: other.get(k, 0) + (edges.get(k, 0) - other.get(k, 0)) * ease(t)
                 for k in keys}
        maxn = other and max(MAXN[mi], MAXN[oi]) or maxn
    ranked = sorted(edges.items(), key=lambda kv: -kv[1])
    for rank, ((a, b), n) in enumerate(ranked):
        if n < 1.2 or a not in pts or b not in pts:
            continue
        pa, pb = pts[a], pts[b]
        ea = min(pa['alpha'], pb['alpha']) * alpha
        gr = ease(min(max(growth * len(ranked) - rank, 0), 1))   # staggered draw-in
        if ea * gr <= 0.02:
            continue
        w = (1.2 + (n / maxn) * 10) * gr
        al = (0.16 + 0.55 * (n / maxn)) * ea * gr * (1 + pulse * 0.12)
        arts.append(ax.plot([pa['x'], pb['x']], [pa['y'], pb['y']], color=WHITE,
                            lw=w * SCALE, alpha=min(al, .9),
                            solid_capstyle='round', zorder=3)[0])
    for name, p in pts.items():
        a = p['alpha'] * alpha
        if a <= 0.02:
            continue
        arts.append(ax.add_patch(Circle((p['x'], p['y']), p['r'], fc=RED, ec=WHITE,
                                        lw=2.2 * SCALE, alpha=a, zorder=5)))
        arts += txt(p['x'], p['y'] - p['r'] * 0.04, str(p['n']['jersey']),
                    p['r'] * 0.95, ha='center', va='center', alpha=a, zorder=7)
        arts += txt(p['x'], p['y'] + p['r'] + 4, p['n']['name'], 15,
                    ha='center', alpha=a * 0.95, zorder=7)
    return arts

def draw_chrome(alpha=1.0):
    arts = []
    for k, c in enumerate([BLUE, WHITE, RED]):
        arts.append(ax.add_patch(Rectangle((k * W / 3, 0), W / 3, 7, fc=c,
                                           ec='none', alpha=alpha)))
    arts += txt(70, 30, '> FRANCE_PASSING_NETWORKS // WC 2022', 19, RED, alpha=alpha)
    arts += txt(W - 70, 30, 'CALCIO AC', 19, TINT, ha='right', alpha=alpha)
    arts += txt(70, H - 46, 'calcioac.com · data: StatsBomb', 16,
                TINT, weight='normal', alpha=alpha * 0.9)
    return arts

def draw_progress(mi, frac, alpha=1.0):
    arts = []
    bw, gap = 96, 12
    x0 = W - 70 - N * bw - (N - 1) * gap
    for k in range(N):
        fill = 1.0 if k < mi else (frac if k == mi else 0.0)
        x = x0 + k * (bw + gap)
        arts.append(ax.add_patch(Rectangle((x, H - 44), bw, 6, fc=BLUE, ec='none',
                                           alpha=0.5 * alpha)))
        if fill > 0:
            arts.append(ax.add_patch(Rectangle((x, H - 44), bw * fill, 6, fc=RED,
                                               ec='none', alpha=alpha)))
    return arts

def wrap_panel(mi, alpha=1.0, t_in=1.0):
    m = DATA[mi]; st = STATS[m['opp']]
    hub = max(m['nodes'], key=lambda n: n['count'])
    top = m['edges'][0]
    arts = []
    X = PANEL_X
    arts += txt(X, 200, m['stage'].upper() + ' · ' + m['date'], 19, TINT, alpha=alpha)
    arts += txt(X, 238, m['score'], 40, WHITE, alpha=alpha)
    rows = [('POSSESSION', f"{st['poss']:.0f}%"),
            ('PASSES', f"{m['completed']}/{m['attempted']} completed"),
            ('xG', st['xg']),
            ('HUB', f"{hub['name']} ({hub['count']})"),
            ('TOP LINK', f"{top['a']} ↔ {top['b']} ({top['n']})")]
    y = 330
    for k, v in rows:
        arts += txt(X, y, k, 16, RED, alpha=alpha)
        arts += txt(X + 215, y, v, 19, WHITE, alpha=alpha)
        y += 52
    ca = ease(min(max((t_in - 0.35) / 0.3, 0), 1)) * alpha
    if ca > 0.02:
        arts.append(ax.add_patch(Rectangle((X, 640), 6, 96, fc=RED, ec='none', alpha=ca)))
        for li, line in enumerate(CALLOUTS[m['opp']]):
            arts += txt(X + 26, 648 + li * 46, line, 19, WHITE if li == 0 else TINT,
                        alpha=ca)
    return arts

# ---------------------------------------------------------------- scenes
def scene_title(t):
    a = ease(min(t / 1.2, 1)) * (1 if t < TITLE_S - 0.8 else max(0, (TITLE_S - t) / 0.8))
    arts = draw_particles(t, a)
    arts += txt(W / 2, 360, 'FRANCE', 110, WHITE, ha='center', alpha=a)
    arts += txt(W / 2, 500, 'A WORLD CUP IN PASSES', 44, RED, ha='center',
                alpha=a * ease(min(max((t - 0.6) / 1.0, 0), 1)))
    arts += txt(W / 2, 580, 'QATAR 2022 · SEVEN GAMES · SEVEN NETWORKS', 22, TINT,
                ha='center', alpha=a * ease(min(max((t - 1.1) / 1.0, 0), 1)))
    bw = ease(min(max((t - 0.3) / 1.2, 0), 1)) * 560
    for k, c in enumerate([BLUE, WHITE, RED]):
        arts.append(ax.add_patch(Rectangle((W / 2 - bw / 2 + k * bw / 3, 640),
                                           bw / 3, 8, fc=c, ec='none', alpha=a)))
    arts += draw_chrome(a * 0.9)
    return arts

def scene_match(mi, t, gt):
    """t in [0, TRANS_S + MATCH_S); first TRANS_S seconds morph from mi-1."""
    arts = draw_particles(gt)
    zoom = 1 + 0.022 * ease(min(max((t - TRANS_S) / MATCH_S, 0), 1))
    cx, cy = PX + PW / 2, PY + PH / 2
    ax.set_xlim(cx - (cx - 0) / zoom * 1, cx + (W - cx) / zoom)
    ax.set_ylim(cy + (H - cy) / zoom, cy - cy / zoom)
    arts += draw_pitch()
    if t < TRANS_S and mi > 0:
        tt = t / TRANS_S
        pts = lerp_layout(mi - 1, mi, tt)
        arts += draw_network(mi, pts, growth=1.0, edge_blend=(mi - 1, tt))
        pa = ease(min(tt / 0.6, 1))
        arts += wrap_panel(mi, alpha=pa, t_in=0)
    else:
        tt = (t - TRANS_S) / MATCH_S if mi > 0 else min(t / MATCH_S, 1)
        fade = ease(min(t / 1.0, 1)) if mi == 0 else 1.0
        growth = min((t if mi == 0 else t - TRANS_S) / 1.1 + 0.15, 1.0)
        pulse = math.sin(gt * 2.2)
        arts += draw_network(mi, lerp_layout(mi, mi, 1), alpha=fade,
                             growth=growth, pulse=pulse)
        arts += wrap_panel(mi, alpha=fade, t_in=tt)
    arts += draw_chrome()
    arts += draw_progress(mi, min(max((t - TRANS_S) / MATCH_S, 0), 1))
    return arts

def scene_finale(t, gt):
    a = ease(min(t / 1.0, 1))
    out = 1.0 if t < FINALE_S - 1.4 else max(0, (FINALE_S - t) / 1.4)
    arts = draw_particles(gt, a * out)
    arts += txt(W / 2, 70, 'SEVEN GAMES. FOUR DIFFERENT FRANCES.', 38, WHITE,
                ha='center', alpha=a * out)
    cols, rows_n = 4, 2
    mw, mh = 380, 254
    gx = (W - cols * mw - (cols - 1) * 36) / 2
    for k in range(N):
        ka = ease(min(max((t - 0.5 - k * 0.22) / 0.5, 0), 1)) * out
        if ka <= 0.02:
            continue
        r, c = divmod(k, cols)
        x0 = gx + c * (mw + 36) + (0 if r == 0 else (mw + 36) / 2)
        y0 = 160 + r * (mh + 130)
        arts += draw_pitch(alpha=ka * 0.8, px=x0, py=y0, pw=mw, ph=mh, lw=1.0)
        m = DATA[k]; maxc = max(n['count'] for n in m['nodes'])
        for e in m['edges'][:14]:
            na = next(n for n in m['nodes'] if n['name'] == e['a'])
            nb = next(n for n in m['nodes'] if n['name'] == e['b'])
            arts.append(ax.plot([x0 + na['x'] / 120 * mw, x0 + nb['x'] / 120 * mw],
                                [y0 + na['y'] / 80 * mh, y0 + nb['y'] / 80 * mh],
                                color=WHITE, lw=(0.5 + e['n'] / MAXN[k] * 3.2) * SCALE,
                                alpha=0.5 * ka)[0])
        for n in m['nodes']:
            arts.append(ax.add_patch(Circle((x0 + n['x'] / 120 * mw,
                                             y0 + n['y'] / 80 * mh),
                                            4 + 5 * n['count'] / maxc,
                                            fc=RED, ec='none', alpha=ka, zorder=5)))
        arts += txt(x0 + mw / 2, y0 + mh + 10, m['opp'].upper(), 17, WHITE,
                    ha='center', alpha=ka)
        arts += txt(x0 + mw / 2, y0 + mh + 36, f"{STATS[m['opp']]['poss']:.0f}% ball",
                    14, TINT, ha='center', alpha=ka)
    sa = ease(min(max((t - 3.4) / 0.8, 0), 1)) * out
    arts += txt(W / 2, H - 168, 'From 62% of the ball to 38%. From control to counter.',
                24, WHITE, ha='center', alpha=sa)
    arts += txt(W / 2, H - 126, 'Same badge, four different teams — one kick short.',
                24, RED, ha='center', alpha=sa)
    arts += draw_chrome(a * out)
    return arts

# ---------------------------------------------------------------- timeline
TOTAL_S = TITLE_S + N * (TRANS_S + MATCH_S) - TRANS_S + FINALE_S   # 1st match has no morph-in
FRAMES = int(TOTAL_S * FPS)
print(f'duration {TOTAL_S:.1f}s · {FRAMES} frames @ {FPS}fps · scale {SCALE}x')

def frame_state(f):
    gt = f / FPS
    if gt < TITLE_S:
        return ('title', gt)
    t = gt - TITLE_S
    block = TRANS_S + MATCH_S
    if t < MATCH_S:                       # first match: no morph
        return ('match', 0, t + TRANS_S)
    t -= MATCH_S
    mi = 1 + int(t // block)
    if mi < N:
        return ('match', mi, t - (mi - 1) * block)
    return ('finale', t - (N - 1) * block)

# ---------------------------------------------------------------- render
os.makedirs('videos', exist_ok=True)
out_path = 'videos/preview.mp4' if PREVIEW else 'videos/france_networks_4k_silent.mp4'
size = (int(1920 * SCALE), int(1080 * SCALE))
ff = imageio_ffmpeg.get_ffmpeg_exe()
enc = subprocess.Popen(
    [ff, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{size[0]}x{size[1]}',
     '-r', str(FPS / STEP if PREVIEW else FPS), '-i', '-',
     '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
     '-pix_fmt', 'yuv420p', out_path],
    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for f in range(0, FRAMES, STEP):
    st = frame_state(f)
    gt = f / FPS
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    if st[0] == 'title':
        arts = scene_title(st[1])
    elif st[0] == 'match':
        arts = scene_match(st[1], st[2], gt)
    else:
        arts = scene_finale(st[1], gt)
    fig.canvas.draw()
    enc.stdin.write(fig.canvas.buffer_rgba())
    for a in arts:
        a.remove()
    if f % (FPS * 5) == 0:
        print(f'  {gt:5.1f}s / {TOTAL_S:.0f}s', flush=True)
enc.stdin.close(); enc.wait()
print('video:', out_path)

# ---------------------------------------------------------------- audio
if not PREVIEW:
    sr = 44100
    n = int(TOTAL_S * sr)
    ts = np.arange(n) / sr
    swell = 0.5 + 0.5 * np.sin(2 * np.pi * ts / 16 - np.pi / 2)
    drone = (0.42 * np.sin(2 * np.pi * 55 * ts) + 0.30 * np.sin(2 * np.pi * 110 * ts)
             + 0.16 * np.sin(2 * np.pi * 165 * ts) + 0.10 * np.sin(2 * np.pi * 220 * ts))
    drone *= 0.35 + 0.3 * swell
    bpm, beat = 84, 60 / 84
    kick = np.zeros(n)
    for b in np.arange(0, TOTAL_S, beat):
        i = int(b * sr); d = np.arange(min(int(0.16 * sr), n - i))
        kick[i:i + len(d)] += np.sin(2 * np.pi * (52 - 90 * d / sr) * d / sr) * np.exp(-d / (0.045 * sr)) * 0.5
    ping = np.zeros(n)
    marks = [TITLE_S] + [TITLE_S + MATCH_S + k * (TRANS_S + MATCH_S) for k in range(N)]
    for b in marks:
        i = int(b * sr)
        if i >= n: continue
        d = np.arange(min(int(1.2 * sr), n - i))
        ping[i:i + len(d)] += np.sin(2 * np.pi * 880 * d / sr) * np.exp(-d / (0.25 * sr)) * 0.10
    audio = drone + kick + ping
    fade = int(2.5 * sr)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    audio = (audio / np.max(np.abs(audio)) * 0.5 * 32767).astype(np.int16)
    with wave.open('videos/score.wav', 'wb') as wv:
        wv.setnchannels(1); wv.setsampwidth(2); wv.setframerate(sr)
        wv.writeframes(audio.tobytes())
    final = 'videos/france_networks_4k.mp4'
    subprocess.run([ff, '-y', '-i', out_path, '-i', 'videos/score.wav',
                    '-c:v', 'copy', '-c:a', 'aac', '-b:a', '160k', '-shortest', final],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(out_path); os.remove('videos/score.wav')
    print('final:', final)
