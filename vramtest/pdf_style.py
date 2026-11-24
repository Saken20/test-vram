"""Shared PDF styling (fonts, colours, tables, KPI tiles, callouts). Requires reportlab."""
import glob
import importlib.util
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import KeepTogether, Spacer  # noqa: F401 — re-exported for report builders
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle

# A Unicode TTF font (Latin + Cyrillic + symbols): VRAMTEST_FONT / VRAMTEST_FONT_BOLD, or searched on the system.
# Without one, the built-in Helvetica is used (Latin-1 only).
FONT_PAIRS = [
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
    ("Ubuntu-R.ttf", "Ubuntu-B.ttf"),
]


def _font_dirs():
    dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts")]
    if os.environ.get("CONDA_PREFIX"):
        dirs.append(os.path.join(os.environ["CONDA_PREFIX"], "fonts"))
    spec = importlib.util.find_spec("matplotlib")  # matplotlib ships DejaVu Sans
    if spec and spec.submodule_search_locations:
        dirs.append(os.path.join(spec.submodule_search_locations[0], "mpl-data", "fonts", "ttf"))
    return dirs


def find_fonts():
    env = os.environ.get("VRAMTEST_FONT"), os.environ.get("VRAMTEST_FONT_BOLD")
    if env[0]:
        return env[0], env[1] or env[0]
    for regular, bold in FONT_PAIRS:
        for d in _font_dirs():
            found = glob.glob(os.path.join(d, "**", regular), recursive=True)
            if found:
                bold_path = os.path.join(os.path.dirname(found[0]), bold)
                return found[0], bold_path if os.path.exists(bold_path) else found[0]
    return None, None


_regular, _bold = find_fonts()
if _regular:
    pdfmetrics.registerFont(TTFont("Sans", _regular))
    pdfmetrics.registerFont(TTFont("Sans-Bold", _bold))
    pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold")
    FONT, FONT_BOLD = "Sans", "Sans-Bold"
else:  # no TTF found: fine for English text (install fonts-dejavu-core or set VRAMTEST_FONT for more glyphs)
    FONT, FONT_BOLD = "Helvetica", "Helvetica-Bold"

INK = colors.HexColor("#1F2937")
MUTED = colors.HexColor("#6B7280")
ACCENT = colors.HexColor("#1D4ED8")
ACCENT_LIGHT = colors.HexColor("#EFF4FF")
LINE = colors.HexColor("#E5E7EB")
ZEBRA = colors.HexColor("#F9FAFB")
WARN_BG = colors.HexColor("#FFF7ED")
WARN = colors.HexColor("#C2410C")
GOOD = colors.HexColor("#047857")


S = {
    "title": ParagraphStyle("title", fontName=FONT_BOLD, fontSize=22, leading=27, textColor=INK),
    "subtitle": ParagraphStyle("subtitle", fontName=FONT, fontSize=10.5, leading=15, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=14, leading=18, textColor=ACCENT,
                         spaceBefore=14, spaceAfter=6),
    "body": ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=14, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName=FONT, fontSize=8, leading=11, textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName=FONT, fontSize=8.5, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName=FONT_BOLD, fontSize=8.5, leading=11, textColor=INK),
    "head": ParagraphStyle("head", fontName=FONT_BOLD, fontSize=8.5, leading=11, textColor=colors.white),
    "kpi_num": ParagraphStyle("kpi_num", fontName=FONT_BOLD, fontSize=18, leading=22, textColor=ACCENT),
    "kpi_lbl": ParagraphStyle("kpi_lbl", fontName=FONT, fontSize=8.5, leading=11, textColor=INK),
    "kpi_sub": ParagraphStyle("kpi_sub", fontName=FONT, fontSize=7.5, leading=10, textColor=MUTED),
}

W = A4[0] - 36 * mm  # text width


def P(text, style="body"):
    return Paragraph(text, S[style])


def table(rows, widths, bold_last=False, highlight_rows=(), align_right_from=1):
    data = [[P(c, "head") for c in rows[0]]]
    for i, r in enumerate(rows[1:], start=1):
        style = "cellb" if (bold_last and i == len(rows) - 1) or i in highlight_rows else "cell"
        data.append([P(str(c), style) for c in r])
    t = Table(data, colWidths=[w * W for w in widths], repeatRows=1, hAlign="LEFT")
    cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    for i in highlight_rows:
        cmds.append(("BACKGROUND", (0, i), (-1, i), ACCENT_LIGHT))
    if bold_last:
        cmds.append(("BACKGROUND", (0, len(data) - 1), (-1, len(data) - 1), ACCENT_LIGHT))
        cmds.append(("LINEABOVE", (0, len(data) - 1), (-1, len(data) - 1), 0.8, ACCENT))
    t.setStyle(TableStyle(cmds))
    return t


def kpis(items):
    cells = []
    for num, label, sub in items:
        cells.append([P(num, "kpi_num"), P(label, "kpi_lbl"), P(sub, "kpi_sub")])
    inner = [Table([[c] for c in cell], colWidths=[W / len(items) - 8]) for cell in cells]
    for t in inner:
        t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    t = Table([inner], colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_LIGHT),
        ("LINEBEFORE", (1, 0), (-1, -1), 2, colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def callout(text, bg=WARN_BG, bar=WARN):
    t = Table([[P(text)]], colWidths=[W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, bar),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


FOOTER_TEXT = ""


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, FOOTER_TEXT)
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
    canvas.setStrokeColor(LINE)
    canvas.line(18 * mm, 13 * mm, A4[0] - 18 * mm, 13 * mm)
    canvas.restoreState()


def build(out, story, title, footer_text):
    global FOOTER_TEXT
    FOOTER_TEXT = footer_text
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=18 * mm, title=title, author="vramtest")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out
