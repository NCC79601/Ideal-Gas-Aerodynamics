import json
import os
from PIL import Image, ImageDraw

with open('./config/config.json', 'r') as f:
    config = json.load(f)

walls = config['walls']
width  = config['map_size'][0]
height = config['map_size'][1]

img = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(img)

for wall in walls:
    draw.line([*wall[0], *wall[1]], fill='black', width=wall[2])

# 反转y轴
img = img.transpose(Image.FLIP_TOP_BOTTOM)

img.show()