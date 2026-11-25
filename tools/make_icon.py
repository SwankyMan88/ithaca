"""Draws Ithaca's app icon (the sidebar logo: a purple-to-blue rounded square with a white four-point sparkle)
and writes src/ithaca.ico with 16-256 px sizes. Run: python tools/make_icon.py"""
import math, os
from PIL import Image, ImageDraw

def icon(size):
    s = size * 4   # draw large, then scale down for clean edges
    img = Image.new('RGBA', (s, s), (0, 0, 0, 0))
    grad = Image.new('RGBA', (s, s))
    a, b = (139, 125, 255), (56, 189, 248)   # the page's --grad
    px = grad.load()
    for y in range(s):
        for x in range(s):
            t = (x + y) / (2 * s - 2)
            px[x, y] = tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3)) + (255,)
    mask = Image.new('L', (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, s - 1, s - 1), radius=int(s * 0.23), fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    cx, cy, r = s * 0.5, s * 0.5, s * 0.30
    pts = []
    for k in range(8):   # four long points, four short ones between them
        ang = -math.pi / 2 + k * math.pi / 4
        rr = r if k % 2 == 0 else r * 0.28
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    d.polygon(pts, fill=(255, 255, 255, 255))
    small = (s * 0.73, s * 0.27, s * 0.085)   # the little second sparkle
    pts = []
    for k in range(8):
        ang = -math.pi / 2 + k * math.pi / 4
        rr = small[2] if k % 2 == 0 else small[2] * 0.3
        pts.append((small[0] + rr * math.cos(ang), small[1] + rr * math.sin(ang)))
    d.polygon(pts, fill=(255, 255, 255, 235))
    return img.resize((size, size), Image.LANCZOS)

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
big = icon(256)
big.save(os.path.join(root, 'src', 'ithaca.ico'), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
big.save(os.path.join(root, 'docs', 'icon.png'))
print('wrote src/ithaca.ico and docs/icon.png')
