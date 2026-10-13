"""Text-and-bubble layout for the example figures (Figure 1, the depression exhibit).

Draws in an axes that covers the whole figure with data units in inches (xlim 0..W, ylim 0..H), so
circles stay circles and text widths can be measured exactly. Rich text = list of (string, bold) segments;
words are wrapped by measured width, so bold phrases can sit mid-sentence. One font family (figstyle's
DejaVu Sans), no italics.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, Polygon

import figstyle as fs

USER_FC, USER_EC = "#eef2f7", "#8aa0b8"
BOT_FC, BOT_EC = "#f3f3f3", "#9a9a9a"
BAD_FC, BAD_EC = "#fbecea", fs.ACCENT
INK = "#222222"
MUTED = "#555555"


class Canvas:
    def __init__(self, width, height):
        fs.setup()
        self.W, self.H = width, height
        self.fig = plt.figure(figsize=(width, height))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()

    def _w(self, s, size, bold):
        t = self.ax.text(0, 0, s, fontsize=size, fontweight="bold" if bold else "normal")
        w = t.get_window_extent(self.r).width / self.fig.dpi
        t.remove()
        return w

    def rich(self, x, y, segments, width, size=7.5, color=INK, leading=1.32, align="left", draw=True):
        """Lay out rich text with its top-left at (x, y) inside `width` inches. Returns (lines, height).
        Words are maximal runs of non-space characters; a word may mix bold and regular runs
        (e.g. a bold phrase followed by a period), and its runs are drawn adjacent with no gap."""
        chars = [(ch, b) for text, b in segments for ch in text]
        words, cur = [], []
        for ch, b in chars + [(" ", False)]:
            if ch.isspace():
                if cur:
                    runs = []
                    for c, bb in cur:
                        if runs and runs[-1][1] == bb:
                            runs[-1][0] += c
                        else:
                            runs.append([c, bb])
                    words.append(runs)
                    cur = []
            else:
                cur.append((ch, b))
        space = self._w("a a", size, False) - self._w("aa", size, False)
        measured = [[(t, b, self._w(t, size, b)) for t, b in runs] for runs in words]
        lines, line, line_w = [], [], 0.0
        for runs in measured:
            ww = sum(r[2] for r in runs)
            add = ww if not line else space + ww
            if line and line_w + add > width:
                lines.append((line, line_w))
                line, line_w = [runs], ww
            else:
                line.append(runs)
                line_w += add
        if line:
            lines.append((line, line_w))
        line_h = size / 72 * leading
        if draw:
            for k, (ws, lw) in enumerate(lines):
                xx = x + (width - lw) / 2 if align == "center" else x
                yy = y - (k + 0.82) * line_h
                for runs in ws:
                    for t, b, rw in runs:
                        self.ax.text(xx, yy, t, fontsize=size, fontweight="bold" if b else "normal",
                                     color=color, va="baseline")
                        xx += rw
                    xx += space
        return len(lines), len(lines) * line_h

    def bubble(self, x, y, width, segments, fc, ec, size=7.5, pad=0.06, tail="left"):
        """Speech bubble with top-left at (x, y); text wraps inside. Tail points left toward an avatar.
        Returns the bubble's bottom y."""
        n, h = self.rich(0, 0, segments, width - 2 * pad, size=size, draw=False)
        bh = h + 2 * pad
        self.ax.add_patch(FancyBboxPatch((x, y - bh), width, bh, boxstyle="round,pad=0,rounding_size=0.06",
                                         fc=fc, ec=ec, lw=0.8, zorder=2))
        if tail == "left":
            ty = y - min(0.16, bh / 2)
            self.ax.add_patch(Polygon([[x + 0.01, ty + 0.05], [x - 0.09, ty - 0.01], [x + 0.01, ty - 0.03]],
                                      closed=True, fc=fc, ec=ec, lw=0.8, zorder=1))
            # cover the tail's seam with the bubble fill
            self.ax.add_patch(Polygon([[x + 0.012, ty + 0.043], [x + 0.012, ty - 0.023], [x + 0.03, ty + 0.01]],
                                      closed=True, fc=fc, ec="none", zorder=3))
        self.rich(x + pad, y - pad, segments, width - 2 * pad, size=size)
        return y - bh

    def user(self, x, y, s=0.22):
        """Flat person icon centred at (x, y)."""
        c = "#8aa0b8"
        self.ax.add_patch(Circle((x, y + s * 0.18), s * 0.2, fc=c, ec="none"))
        self.ax.add_patch(Ellipse((x, y - s * 0.27), s * 0.62, s * 0.42, fc=c, ec="none"))

    def bot(self, x, y, s=0.22, bad=False):
        """Flat robot-head icon centred at (x, y); small horns when `bad`."""
        c = fs.ACCENT if bad else "#8c8c8c"
        if bad:
            for sg in (-1, 1):
                self.ax.add_patch(Polygon([[x + sg * s * 0.36, y + s * 0.22], [x + sg * s * 0.16, y + s * 0.26],
                                           [x + sg * s * 0.40, y + s * 0.50]], closed=True, fc=c, ec="none"))
        else:
            self.ax.plot([x, x], [y + s * 0.24, y + s * 0.40], color=c, lw=1.0, solid_capstyle="round")
            self.ax.add_patch(Circle((x, y + s * 0.43), s * 0.06, fc=c, ec="none"))
        self.ax.add_patch(FancyBboxPatch((x - s * 0.36, y - s * 0.30), s * 0.72, s * 0.56,
                                         boxstyle=f"round,pad=0,rounding_size={s * 0.12}", fc=c, ec="none"))
        for dx in (-s * 0.14, s * 0.14):
            self.ax.add_patch(Circle((x + dx, y + s * 0.0), s * 0.065, fc="white", ec="none"))

    def text(self, x, y, s, size=7.5, bold=False, color=INK, ha="left", va="top"):
        return self.ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", color=color,
                            ha=ha, va=va)

    def arrow(self, x0, y0, x1, y1):
        self.ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#444444", mutation_scale=10))

    def save(self, path):
        """Save PDF + PNG without bbox cropping, so in-figure coordinates stay exact (overpic links)."""
        import matplotlib as mpl
        with mpl.rc_context({"savefig.bbox": "standard"}):  # figstyle sets "tight"; keep the exact canvas
            self.fig.savefig(path + ".pdf", pad_inches=0)
            self.fig.savefig(path + ".png", dpi=200, pad_inches=0)
        lo = min(t.get_window_extent(self.r).y0 for t in self.ax.texts) / self.fig.dpi
        assert lo >= -0.005, f"content runs {-lo:.2f} in below the canvas; increase the height"
        plt.close(self.fig)


def segs(text, bold_phrases):
    """Split `text` into (segment, bold) pieces, bolding each phrase in `bold_phrases` (must occur verbatim)."""
    pieces = [(text, False)]
    for ph in bold_phrases:
        assert ph in text, f"bold phrase not found verbatim: {ph!r}"
        out = []
        for s, b in pieces:
            if b or ph not in s:
                out.append((s, b))
                continue
            i = s.index(ph)
            if s[:i]:
                out.append((s[:i], False))
            out.append((ph, True))
            if s[i + len(ph):]:
                out.append((s[i + len(ph):], False))
        pieces = out
    return pieces
