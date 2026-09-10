from bs4 import BeautifulSoup
import json


HTML_PATH = "data/aorus_master_16_am6h.html"
OUTPUT_PATH = "data/specs.json"


SPEC_KEYS = [
    "作業系統",
    "中央處理器",
    "顯示晶片",
    "顯示器",
    "記憶體",
    "儲存裝置",
    "鍵盤種類",
    "連接埠",
    "音效",
    "通訊",
    "視訊鏡頭",
    "安全裝置",
    "電池",
    "變壓器",
    "尺寸",
    "重量",
    "顏色",
]


with open(HTML_PATH, "r", encoding="utf-8") as file:
    html = file.read()


soup = BeautifulSoup(html, "html.parser")

text = soup.get_text("\n", strip=True)

lines = [line.strip() for line in text.splitlines() if line.strip()]


# 找出真正產品規格區的開始位置
start_index = None

for i in range(len(lines) - 1):
    if lines[i] == "產品規格" and lines[i + 1] == "作業系統":
        start_index = i + 1
        break


if start_index is None:
    raise ValueError("找不到產品規格區")


specs = {}

current_key = None


for line in lines[start_index:]:

    # 遇到下一個規格 key
    if line in SPEC_KEYS:
        current_key = line

        if current_key not in specs:
            specs[current_key] = []

        continue

    # 已經進入某個規格欄位
    if current_key is not None:
        specs[current_key].append(line)

    # 第一組規格在 Dark Tide 結束
    if current_key == "顏色" and line == "Dark Tide":
        break


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(specs, file, ensure_ascii=False, indent=2)


print("規格解析完成")
print(f"已輸出到：{OUTPUT_PATH}")

for key, values in specs.items():
    print()
    print("###", key)

    for value in values:
        print("-", value)