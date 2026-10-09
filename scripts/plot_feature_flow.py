"""
================================================================================
特征流可视化: (d) 从 (b) 右侧引出, 实心一体箭头, 拐弯箭头绕过 (c)

布局:
        (a)
         ↓  Processed by 1D convolutional blocks
        (b) ───────────────╮
         ↓  Processed by the│  Processed by the SCTM
            Fourier module  │
        (c)                 │
                            │
            (d) ←───────────╯
================================================================================
用法: py plot_feature_flow.py
================================================================================
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, 'docs', 'figures', 'feature_flow')
OUT = os.path.join(FIGDIR, 'feature_flow_combined.png')

IMG_A = os.path.join(FIGDIR, 'aaa_input.png')
IMG_B = os.path.join(FIGDIR, 'aaa_after_1d.png')
IMG_C = os.path.join(FIGDIR, 'aaa_after_sctm.png')
IMG_D = os.path.join(FIGDIR, 'aaa_after_pa_50.png')


def load_resize(path, target_w, h_scale=1.0):
    img = Image.open(path).convert('RGB')
    ratio = target_w / img.width
    target_h = int(img.height * ratio * h_scale)   # h_scale 拉长高度
    return img.resize((target_w, target_h), Image.LANCZOS)


def solid_arrow(draw, x, y1, y2, shaft_w=34, head_w=130, head_h=70, color='black'):
    """实心一体箭头 (杆+头同色, 无缝)"""
    head_top = y2 - head_h
    half = shaft_w // 2
    draw.rectangle([x - half, y1, x + half, head_top], fill=color)
    draw.polygon([(x - head_w // 2, head_top),
                  (x + head_w // 2, head_top),
                  (x, y2)], fill=color)


def route_arrow_left(draw, x_from, y_from, x_route, y_target,
                     shaft_w=34, head_w=140, head_h=110, color='black'):
    """实心拐弯箭头: 右 -> 下 -> 左箭头(指向 d 右侧), 一体无缝"""
    half = shaft_w // 2
    # 水平向右段 (从 b 右缘)
    draw.rectangle([x_from - half, y_from - half, x_route + half, y_from + half], fill=color)
    # 竖直向下段
    draw.rectangle([x_route - half, y_from - half, x_route + half, y_target + half], fill=color)
    # 水平向左回段 (到箭头头底)
    draw.rectangle([x_from + head_h - half, y_target - half, x_route + half, y_target + half], fill=color)
    # 箭头头 (指向左, 顶点在 x_from = d 右缘)
    draw.polygon([
        (x_from, y_target),                        # 左顶点 (指向 d)
        (x_from + head_h, y_target - head_w // 2),  # 右上
        (x_from + head_h, y_target + head_w // 2),  # 右下
    ], fill=color)


def draw_text(draw, x, y, text, font, anchor_left=True):
    tw = draw.textlength(text, font=font) if hasattr(draw, 'textlength') else len(text) * 22
    tx = x if anchor_left else x - tw
    draw.text((tx, y), text, fill='black', font=font)


def draw_text_center(draw, cx, y, text, font):
    tw = draw.textlength(text, font=font) if hasattr(draw, 'textlength') else len(text) * 22
    draw.text((cx - tw // 2, y), text, fill='black', font=font)


def draw_text_vertical(canvas, x, y, text, font, angle=-90):
    """竖排文字: 渲染后旋转粘贴到画布 (angle=-90 使 P 在上, 自上而下读)"""
    tmp = Image.new('RGB', (len(text) * 140, 320), 'white')
    td = ImageDraw.Draw(tmp)
    td.text((10, 10), text, fill='black', font=font)
    bbox = tmp.getbbox()
    if bbox:
        tmp = tmp.crop(bbox)
    tmp = tmp.rotate(angle, expand=True, fillcolor='white')
    canvas.paste(tmp, (x, y))


def load_fonts():
    for path in ['C:/Windows/Fonts/msyhbd.ttc',   # 微软雅黑粗体
                 'C:/Windows/Fonts/simhei.ttf',
                 'C:/Windows/Fonts/msyh.ttc']:
        try:
            return (ImageFont.truetype(path, 90), ImageFont.truetype(path, 120))
        except Exception:
            continue
    return (ImageFont.load_default(), ImageFont.load_default())


def main():
    font_big, font_arrow = load_fonts()

    # 子图宽度 (拉大拉长)
    sub_w = 6400
    h_scale = 1.6           # 高度拉伸倍数 (拉长)
    a = load_resize(IMG_A, sub_w, h_scale)
    b = load_resize(IMG_B, sub_w, h_scale)
    c = load_resize(IMG_C, sub_w, h_scale)
    d = load_resize(IMG_D, sub_w, h_scale)

    margin_x = 150
    margin_y = 100
    route_w = 160           # 右侧拐弯通道宽度
    arrow_gap = 280         # 普通箭头区域高度 (留足箭杆空间)

    img_x = margin_x
    x_center = img_x + sub_w // 2
    x_route = img_x + sub_w + route_w          # 拐弯通道竖线位置

    canvas_w = img_x + sub_w + route_w + 300   # 右侧只留竖排文字所需空间
    label_h = 80            # (d) 下方标签预留空间
    total_h = (margin_y + a.height + arrow_gap + b.height + arrow_gap
               + c.height + arrow_gap + d.height + label_h + margin_y)

    canvas = Image.new('RGB', (canvas_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)

    # ---- (a) ----
    y = margin_y
    canvas.paste(a, (img_x, y))
    y += a.height

    # 标签 (a) 居中, 箭头从标签下方开始 (不重合)
    draw_text_center(draw, x_center, y - 20, '(a)', font_big)
    a1_top, a1_bot = y + 110, y + arrow_gap
    solid_arrow(draw, x_center, a1_top, a1_bot)
    draw_text(draw, x_center + 130, a1_top + 15,
              'Processed by 1D convolutional blocks', font_arrow)
    y = a1_bot

    # ---- (b) ----
    canvas.paste(b, (img_x, y))
    b_middle = y + b.height // 2
    y += b.height

    # 标签 (b) 居中
    draw_text_center(draw, x_center, y - 20, '(b)', font_big)
    a2_top, a2_bot = y + 110, y + arrow_gap
    solid_arrow(draw, x_center, a2_top, a2_bot)
    draw_text(draw, x_center + 130, a2_top + 15,
              'Processed by the Fourier module', font_arrow)
    y = a2_bot

    # ---- (c) ----
    canvas.paste(c, (img_x, y))
    y += c.height

    # 标签 (c) 居中
    draw_text_center(draw, x_center, y - 20, '(c)', font_big)

    # 拐弯箭头: 从 (b) 右缘中间 -> 右通道 -> 指向 (d) 右侧
    d_top = y + arrow_gap
    y_target = d_top + d.height // 2        # (d) 中间高度
    route_arrow_left(draw, img_x + sub_w, b_middle, x_route, y_target)
    # 文字竖排, 左移避免出界
    draw_text_vertical(canvas, x_route + 15, b_middle + 200,
                       'Processed by the SCTM', font_arrow)
    y = d_top

    # ---- (d) ----
    canvas.paste(d, (img_x, y))
    y += d.height
    # 标签 (d) 居中
    draw_text_center(draw, x_center, y - 20, '(d)', font_big)

    canvas.save(OUT)
    print(f"已生成: {OUT}")
    print(f"  画布尺寸: {canvas_w} x {total_h}")


if __name__ == '__main__':
    main()
