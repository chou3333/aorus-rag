from bs4 import BeautifulSoup
import json


HTML_PATH = "data/aorus_master_16_am6h.html"
OUTPUT_PATH = "data/specs.json"


VARIANTS = [
    "AORUS MASTER 16 BZH",
    "AORUS MASTER 16 BYH",
    "AORUS MASTER 16 BXH",
]


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


# 找到「顯示差異」區塊
compare_index = None

for i, line in enumerate(lines):
    if line == "顯示差異":
        compare_index = i
        break

if compare_index is None:
    raise ValueError("找不到『顯示差異』區塊")


# 顯示差異後面應該依序出現 17 個規格欄位名稱
key_start = compare_index + 1
key_end = key_start + len(SPEC_KEYS)

detected_keys = lines[key_start:key_end]

if detected_keys != SPEC_KEYS:
    raise ValueError(
        "規格欄位順序與預期不一致\n"
        f"偵測到：{detected_keys}"
    )


# 真正的三組規格值從欄位名稱之後開始
value_start = key_end

all_specs = {}

current_index = value_start


for variant in VARIANTS:

    specs = {}

    for key_index, key in enumerate(SPEC_KEYS):

        values = []

        # 最後一個欄位「顏色」以 Dark Tide 結束
        if key == "顏色":
            while current_index < len(lines):
                line = lines[current_index]
                values.append(line)
                current_index += 1

                if line == "Dark Tide":
                    break

        else:
            # 下一個欄位的起點無法直接用文字判斷，
            # 因此使用已知規格內容特徵來切分。
            next_key = SPEC_KEYS[key_index + 1]

            while current_index < len(lines):

                line = lines[current_index]

                # 依下一個欄位的特徵判斷切點
                if next_key == "中央處理器" and line.startswith("Intel® Core™"):
                    break

                if next_key == "顯示晶片" and line.startswith("NVIDIA® GeForce"):
                    break

                if next_key == "顯示器" and line.startswith('16" 16:10'):
                    break

                if next_key == "記憶體" and line.startswith("Up to 64GB DDR5"):
                    break

                if next_key == "儲存裝置" and line.startswith("1x PCIe Gen5"):
                    break

                if next_key == "鍵盤種類" and line.startswith("3-zone RGB Backlit Keyboard"):
                    break

                if next_key == "連接埠" and line == "Left Side:":
                    break

                if next_key == "音效" and line.startswith("4x 2W speakers"):
                    break

                if next_key == "通訊" and line.startswith("WIFI 7"):
                    break

                if next_key == "視訊鏡頭" and line.startswith("FHD (1080p) IR Webcam"):
                    break

                if next_key == "安全裝置" and line.startswith("Firmware-based TPM"):
                    break

                if next_key == "電池" and line.startswith("Li-ion 99Wh"):
                    break

                if next_key == "變壓器" and line.startswith("330W AC Adapter"):
                    break

                if next_key == "尺寸" and line.startswith("357 x 254"):
                    break

                if next_key == "重量" and line.startswith("~2.5 kg"):
                    break

                if next_key == "顏色" and line == "Dark Tide":
                    break

                values.append(line)
                current_index += 1

        specs[key] = values

    all_specs[variant] = specs


with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(all_specs, file, ensure_ascii=False, indent=2)


print("規格解析完成")
print(f"已輸出到：{OUTPUT_PATH}")

for variant, specs in all_specs.items():
    print()
    print("=" * 60)
    print(variant)
    print("=" * 60)

    print("GPU:")
    for value in specs["顯示晶片"]:
        print("-", value)

    print("RAM:")
    for value in specs["記憶體"]:
        print("-", value)

    print("Battery:")
    for value in specs["電池"]:
        print("-", value)