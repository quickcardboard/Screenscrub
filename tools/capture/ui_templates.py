"""HTML mock-ups of common screen content, rendered to slides by make_slides.py.

Each template is a function (variant) -> HTML string for a 1920x1080 page. Variants change theme,
font size and accent colour so one template yields several distinct slides. Content is generic
and written for this project (no third-party UI is reproduced).
"""
import random

FONTS = ["Roboto", "DejaVu Sans", "Liberation Sans", "Liberation Serif", "DejaVu Sans Mono"]
ACCENTS = ["#1a73e8", "#d93025", "#188038", "#f29900", "#9334e6", "#007b83", "#e8710a"]
WORDS = ("account settings update review project schedule meeting budget report summary weather "
         "forecast travel booking order delivery payment invoice photo album music playlist video "
         "channel episode season match score league team player result traffic route station train "
         "ticket recipe kitchen garden window storage backup network battery display privacy "
         "security message reply forward archive calendar reminder notes draft share download "
         "Amsterdam Berlin Chicago Denver Edmonton Florence Geneva Houston Istanbul Jakarta Kyoto Lisbon "
         "Madrid Nairobi Oslo Prague Quebec Rome Seoul Toronto Utrecht Vienna Warsaw Zurich "
         "Alice Bruno Chen Divya Emeka Farah Goran Hana Ivan Jonas Keiko Luis Maya Nikos Omar Priya "
         "quarterly invoice#4471 v2.3.1 14:05 98.6% 3,250 km 19.99 #A7F2 Q3-2026 ID-88213 OK 404 "
         "abstract balance cadence diligent elbow fathom glacier harbor island jasmine kettle lantern "
         "meadow nectar orchard pebble quartz ripple saffron timber umbrella velvet walnut yonder zephyr").split()


def words(rnd, n):
    w = [rnd.choice(WORDS) if rnd.random() < .85 else str(rnd.randint(2, 9999)) for _ in range(n)]
    w[0] = w[0].capitalize()
    return " ".join(w)


def theme(dark):
    return dict(bg="#121212" if dark else "#ffffff", panel="#1e1f20" if dark else "#f1f3f4",
                fg="#e8eaed" if dark else "#202124", sub="#9aa0a6" if dark else "#5f6368",
                line="#3c4043" if dark else "#dadce0", chip="#2d2f31" if dark else "#e8f0fe")


def page(body, t, font, size, extra_css=""):
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box}} html,body{{margin:0;width:1920px;height:1080px;overflow:hidden}}
body{{background:{t['bg']};color:{t['fg']};font-family:'{font}',sans-serif;font-size:{size}px}}
.sub{{color:{t['sub']}}} .panel{{background:{t['panel']}}} {extra_css}
</style></head><body>{body}</body></html>"""


def list_app(v):
    rnd = random.Random(v); dark = v % 2 == 1; t = theme(dark)
    acc = ACCENTS[v % len(ACCENTS)]; size = [18, 20, 24, 28][v % 4]
    rows = "".join(
        f"<div style='display:flex;align-items:center;padding:14px 28px;border-bottom:1px solid {t['line']}'>"
        f"<div style='width:44px;height:44px;border-radius:50%;background:{rnd.choice(ACCENTS)};color:#fff;"
        f"display:flex;align-items:center;justify-content:center;margin-right:20px'>{rnd.choice('ABCDEFGHJKLMNPRS')}</div>"
        f"<div style='flex:1'><div style='font-weight:600'>{words(rnd, 3)}</div><div class=sub>{words(rnd, 9)}</div></div>"
        f"<div class=sub>{rnd.randint(1, 12)}:{rnd.randint(10, 59)} {'AM' if rnd.random() < .5 else 'PM'}</div></div>"
        for _ in range(16))
    side = "".join(f"<div style='padding:12px 24px;border-radius:0 24px 24px 0;{'background:'+t['chip'] if i == 0 else ''}'>"
                   f"{words(rnd, 1)}</div>" for i in range(9))
    body = (f"<div style='display:flex;height:100%'><div class=panel style='width:300px;padding-top:90px'>{side}</div>"
            f"<div style='flex:1'><div style='height:72px;display:flex;align-items:center;padding:0 28px;font-size:1.3em'>"
            f"<span style='color:{acc};font-weight:700'>{words(rnd, 1)}</span>&nbsp;{words(rnd, 1)}"
            f"<div class=panel style='margin-left:60px;flex:1;height:48px;border-radius:24px'></div></div>{rows}</div></div>")
    return page(body, t, FONTS[v % 3], size)


def settings(v):
    rnd = random.Random(100 + v); dark = v % 2 == 0; t = theme(dark)
    acc = ACCENTS[(v + 2) % len(ACCENTS)]
    items = ""
    for _ in range(10):
        on = rnd.random() < .5
        items += (f"<div style='display:flex;align-items:center;padding:22px 0;border-bottom:1px solid {t['line']}'>"
                  f"<div style='flex:1'><div style='font-size:1.1em'>{words(rnd, 2)}</div><div class=sub>{words(rnd, 8)}</div></div>"
                  f"<div style='width:64px;height:34px;border-radius:17px;background:{acc if on else t['line']};position:relative'>"
                  f"<div style='position:absolute;top:4px;{'right' if on else 'left'}:4px;width:26px;height:26px;border-radius:50%;background:#fff'></div></div></div>")
    body = (f"<div style='max-width:1100px;margin:0 auto;padding:60px 40px'><h1 style='font-weight:400'>{words(rnd, 2)}</h1>"
            f"<div class=panel style='border-radius:16px;padding:8px 32px'>{items}</div></div>")
    return page(body, t, FONTS[(v + 1) % 3], [20, 24, 28][v % 3])


def document(v):
    rnd = random.Random(200 + v); dark = v % 3 == 2; t = theme(dark)
    paras = "".join((f"<h2 style='margin:28px 0 8px'>{words(rnd, 3)}</h2>" if rnd.random() < .25 else "") +
                    f"<p style='margin:0 0 18px;line-height:1.55'>{words(rnd, rnd.randint(40, 70))}.</p>" for _ in range(8))
    body = (f"<div style='width:1250px;margin:0 auto;padding:50px 60px;height:100%;"
            f"background:{'#1e1f20' if dark else '#ffffff'}'><h1>{words(rnd, 4)}</h1>{paras}</div>")
    t2 = dict(t); t2['bg'] = "#000" if dark else "#f1f3f4"
    return page(body, t2, FONTS[[0, 3, 1, 2][v % 4]], [17, 20, 24, 30][v % 4])


def dashboard(v):
    rnd = random.Random(300 + v); dark = v % 2 == 1; t = theme(dark)
    cards = ""
    for i in range(6):
        acc = ACCENTS[(v + i) % len(ACCENTS)]
        bars = "".join(f"<div style='flex:1;margin:0 3px;background:{acc};height:{rnd.randint(15, 100)}%;border-radius:4px 4px 0 0'></div>" for _ in range(12))
        cards += (f"<div class=panel style='border-radius:16px;padding:24px;display:flex;flex-direction:column'>"
                  f"<div class=sub>{words(rnd, 2)}</div><div style='font-size:2.2em;font-weight:600;margin:8px 0'>"
                  f"{rnd.randint(10, 999)}{rnd.choice(['', '%', 'k', ' GB'])}</div>"
                  f"<div style='flex:1;display:flex;align-items:flex-end'>{bars}</div></div>")
    body = (f"<div style='padding:40px'><h1 style='margin:0 0 30px;font-weight:500'>{words(rnd, 2)}</h1>"
            f"<div style='display:grid;grid-template-columns:repeat(3,1fr);grid-template-rows:400px 400px;gap:24px'>{cards}</div></div>")
    return page(body, t, FONTS[v % 3], 20)


def code_editor(v):
    rnd = random.Random(400 + v); dark = v % 3 != 2; t = theme(dark)
    kw = ["def", "return", "if", "for", "class", "import", "val", "fun", "while", "else"]
    cols = ["#c678dd", "#61afef", "#98c379", "#e5c07b", "#e06c75", "#56b6c2"] if dark else ["#a626a4", "#4078f2", "#50a14f", "#986801", "#e45649", "#0184bc"]
    lines = ""
    for i in range(40):
        ind = "&nbsp;" * (4 * rnd.randint(0, 3))
        toks = " ".join(f"<span style='color:{rnd.choice(cols)}'>{rnd.choice(kw) if rnd.random() < .3 else rnd.choice(WORDS)}</span>"
                        for _ in range(rnd.randint(1, 7)))
        lines += f"<div><span class=sub style='display:inline-block;width:60px;text-align:right;margin-right:24px'>{i + 1}</span>{ind}{toks}</div>"
    side = "".join(f"<div style='padding:4px 16px'>{rnd.choice(WORDS)}.{rnd.choice(['py', 'kt', 'md', 'json'])}</div>" for _ in range(30))
    body = (f"<div style='display:flex;height:100%'><div class=panel style='width:320px;padding-top:20px'>{side}</div>"
            f"<div style='flex:1;padding:20px;line-height:1.45'>{lines}</div></div>")
    return page(body, t, "DejaVu Sans Mono", [16, 18, 22][v % 3])


def chat(v):
    rnd = random.Random(500 + v); dark = v % 2 == 0; t = theme(dark)
    acc = ACCENTS[v % len(ACCENTS)]
    msgs = ""
    for _ in range(9):
        me = rnd.random() < .5
        msgs += (f"<div style='display:flex;justify-content:{'flex-end' if me else 'flex-start'};margin:10px 0'>"
                 f"<div style='max-width:900px;padding:16px 22px;border-radius:22px;"
                 f"background:{acc if me else t['panel']};color:{'#fff' if me else t['fg']}'>{words(rnd, rnd.randint(5, 30))}</div></div>")
    body = f"<div style='max-width:1300px;margin:0 auto;padding:40px'>{msgs}</div>"
    return page(body, t, FONTS[v % 3], [22, 26, 30][v % 3])


def search_results(v):
    rnd = random.Random(600 + v); dark = v % 2 == 1; t = theme(dark)
    link = "#8ab4f8" if dark else "#1a0dab"
    res = "".join(f"<div style='margin:0 0 34px'><div class=sub style='font-size:.85em'>www.{rnd.choice(WORDS)}.example</div>"
                  f"<div style='color:{link};font-size:1.3em;margin:4px 0'>{words(rnd, 6)}</div>"
                  f"<div class=sub style='line-height:1.5'>{words(rnd, 30)}</div></div>" for _ in range(6))
    body = (f"<div style='padding:30px 180px'><div style='display:flex;align-items:center;margin-bottom:40px'>"
            f"<div class=panel style='width:800px;height:52px;border-radius:26px;border:1px solid {t['line']}'></div></div>{res}</div>")
    return page(body, t, FONTS[v % 2], 20)


TEMPLATES = {"list": list_app, "settings": settings, "document": document, "dashboard": dashboard,
             "code": code_editor, "chat": chat, "search": search_results}
