#!/usr/bin/env python3
"""生成 PNG 图标从 SVG - 使用 Brain 图标"""
import sys
from PIL import Image, ImageDraw
import math

def draw_brain_icon(draw, center_x, center_y, size):
    """绘制 Brain 图标"""
    scale = size / 24.0  # 原始图标是 24x24
    
    # Brain 图标路径（基于 Lucide Brain SVG）
    # 左半部分
    left_path = [
        (center_x - 5*scale, center_y - 7*scale),
        (center_x - 8*scale, center_y - 4*scale),
        (center_x - 9*scale, center_y),
        (center_x - 8*scale, center_y + 4*scale),
        (center_x - 5*scale, center_y + 7*scale),
        (center_x - 2*scale, center_y + 8*scale),
        (center_x, center_y + 7*scale),
    ]
    
    # 右半部分
    right_path = [
        (center_x + 5*scale, center_y - 7*scale),
        (center_x + 8*scale, center_y - 4*scale),
        (center_x + 9*scale, center_y),
        (center_x + 8*scale, center_y + 4*scale),
        (center_x + 5*scale, center_y + 7*scale),
        (center_x + 2*scale, center_y + 8*scale),
        (center_x, center_y + 7*scale),
    ]
    
    # 绘制填充的大脑形状
    # 左半部分
    left_points = left_path + [(center_x, center_y)]
    draw.polygon(left_points, fill='white', outline=None)
    
    # 右半部分
    right_points = right_path + [(center_x, center_y)]
    draw.polygon(right_points, fill='white', outline=None)
    
    # 绘制细节线条
    stroke_width = max(1, int(scale * 0.5))
    # 中间连接线
    draw.line([(center_x - 3*scale, center_y + 2*scale), 
                (center_x + 3*scale, center_y + 2*scale)], 
               fill='white', width=stroke_width)
    
    # 顶部细节
    draw.arc([center_x - 2*scale, center_y - 6*scale, 
              center_x + 2*scale, center_y - 2*scale],
             start=0, end=180, fill='white', width=stroke_width)

def generate_png_icons():
    """生成 PNG 图标"""
    sizes = [
        ('icon-light-32x32.png', 32),
        ('icon-dark-32x32.png', 32),
        ('apple-icon.png', 180),
    ]
    
    for filename, size in sizes:
        # 创建渐变背景
        img = Image.new('RGB', (size, size), color='#3B82F6')
        draw = ImageDraw.Draw(img)
        
        # 绘制渐变效果
        for y in range(size):
            # 从蓝色到紫色渐变
            ratio = y / size
            r = int(59 + (139 - 59) * ratio)  # #3B82F6 to #8B5CF6
            g = int(130 + (92 - 130) * ratio)
            b = int(246 + (246 - 246) * ratio)
            draw.line([(0, y), (size, y)], fill=(r, g, b))
        
        # 绘制 Brain 图标
        center_x, center_y = size // 2, size // 2
        draw_brain_icon(draw, center_x, center_y, size * 0.7)
        
        output_path = f'public/{filename}'
        img.save(output_path, 'PNG')
        print(f'Generated {filename} ({size}x{size})')

if __name__ == '__main__':
    try:
        from PIL import Image, ImageDraw
        generate_png_icons()
    except ImportError:
        print("Error: PIL/Pillow not installed")
        print("Install with: pip install Pillow")
        sys.exit(1)
