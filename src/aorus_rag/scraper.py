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


def split_content(value):
    """
    將規格內容中的 <br> 拆成多行。
    最後回傳 list[str]，與原本 specs.json 格式一致。
    """
    if value is None:
        return []

    soup = BeautifulSoup(str(value), "html.parser")

    text = soup.get_text("\n", strip=True)

    return [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]


def resolve_nuxt_value(data, value):
    """
    Nuxt __NUXT_DATA__ 會使用整數 index
    指向 data array 中的其他資料。

    這個函式負責把 index 還原成真正的內容。
    """

    if isinstance(value, int):
        if 0 <= value < len(data):
            return resolve_nuxt_value(
                data,
                data[value],
            )

        return value

    if isinstance(value, list):
        return [
            resolve_nuxt_value(data, item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            key: resolve_nuxt_value(data, item)
            for key, item in value.items()
        }

    return value


def parse_nuxt_specs(html):
    """
    優先嘗試從 __NUXT_DATA__ 中解析結構化規格。

    找不到 Nuxt 資料時回傳 None，
    之後再使用畫面文字解析方式。
    """

    soup = BeautifulSoup(html, "html.parser")

    script = soup.find(
        "script",
        id="__NUXT_DATA__",
    )

    if script is None:
        return None

    raw_json = script.string

    if not raw_json:
        return None

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, list):
        return None

    all_specs = {}

    # Nuxt 的 data array 中可能有很多內容，
    # 我們只找看起來像產品資料的 dict。
    for value in data:

        if not isinstance(value, dict):
            continue

        if (
            "productId" not in value
            or "title" not in value
            or "specItem" not in value
        ):
            continue

        resolved = resolve_nuxt_value(
            data,
            value,
        )

        product_id = str(
            resolved.get("productId", "")
        ).strip()

        title = str(
            resolved.get("title", "")
        ).strip()

        # 只處理這三個型號
        if product_id not in {
            "BZH",
            "BYH",
            "BXH",
        }:
            continue

        product_name = (
            title
            if title
            else f"AORUS MASTER 16 {product_id}"
        )

        specs = {}

        spec_items = resolved.get(
            "specItem",
            [],
        )

        if not isinstance(spec_items, list):
            continue

        for item in spec_items:

            if not isinstance(item, dict):
                continue

            key = str(
                item.get("itemTitle", "")
            ).strip()

            value = item.get(
                "itemContent",
                "",
            )

            if not key:
                continue

            specs[key] = split_content(value)

        all_specs[product_name] = specs

    if not all_specs:
        return None

    # 固定按照 BZH / BYH / BXH 排列，
    # 但資料仍然依產品自己的 productId 對應，
    # 不依賴網頁出現順序。
    ordered_specs = {}

    for variant in VARIANTS:
        if variant in all_specs:
            ordered_specs[variant] = all_specs[variant]

    # 如果頁面包含其他命名方式，也保留
    for product_name, specs in all_specs.items():
        if product_name not in ordered_specs:
            ordered_specs[product_name] = specs

    return ordered_specs


def parse_visible_specs(html):
    """
    備援方式：
    從瀏覽器儲存下來的可見網頁文字中解析規格。
    """

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    text = soup.get_text(
        "\n",
        strip=True,
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # 找到「顯示差異」區塊
    compare_index = None

    for i, line in enumerate(lines):
        if line == "顯示差異":
            compare_index = i
            break

    if compare_index is None:
        raise ValueError(
            "找不到『顯示差異』區塊"
        )

    # 顯示差異後面應該依序出現 17 個規格欄位名稱
    key_start = compare_index + 1
    key_end = (
        key_start
        + len(SPEC_KEYS)
    )

    detected_keys = lines[
        key_start:key_end
    ]

    if detected_keys != SPEC_KEYS:
        raise ValueError(
            "規格欄位順序與預期不一致\n"
            f"偵測到：{detected_keys}"
        )

    value_start = key_end

    all_specs = {}
    current_index = value_start

    for variant in VARIANTS:

        specs = {}

        for key_index, key in enumerate(
            SPEC_KEYS
        ):

            values = []

            # 最後一個欄位「顏色」
            if key == "顏色":

                while current_index < len(lines):

                    line = lines[
                        current_index
                    ]

                    values.append(line)
                    current_index += 1

                    if line == "Dark Tide":
                        break

            else:
                next_key = SPEC_KEYS[
                    key_index + 1
                ]

                while current_index < len(lines):

                    line = lines[
                        current_index
                    ]

                    if (
                        next_key == "中央處理器"
                        and line.startswith(
                            "Intel® Core™"
                        )
                    ):
                        break

                    if (
                        next_key == "顯示晶片"
                        and line.startswith(
                            "NVIDIA® GeForce"
                        )
                    ):
                        break

                    if (
                        next_key == "顯示器"
                        and line.startswith(
                            '16" 16:10'
                        )
                    ):
                        break

                    if (
                        next_key == "記憶體"
                        and line.startswith(
                            "Up to 64GB DDR5"
                        )
                    ):
                        break

                    if (
                        next_key == "儲存裝置"
                        and line.startswith(
                            "1x PCIe Gen5"
                        )
                    ):
                        break

                    if (
                        next_key == "鍵盤種類"
                        and line.startswith(
                            "3-zone RGB Backlit Keyboard"
                        )
                    ):
                        break

                    if (
                        next_key == "連接埠"
                        and line == "Left Side:"
                    ):
                        break

                    if (
                        next_key == "音效"
                        and line.startswith(
                            "4x 2W speakers"
                        )
                    ):
                        break

                    if (
                        next_key == "通訊"
                        and line.startswith(
                            "WIFI 7"
                        )
                    ):
                        break

                    if (
                        next_key == "視訊鏡頭"
                        and line.startswith(
                            "FHD (1080p) IR Webcam"
                        )
                    ):
                        break

                    if (
                        next_key == "安全裝置"
                        and line.startswith(
                            "Firmware-based TPM"
                        )
                    ):
                        break

                    if (
                        next_key == "電池"
                        and line.startswith(
                            "Li-ion 99Wh"
                        )
                    ):
                        break

                    if (
                        next_key == "變壓器"
                        and line.startswith(
                            "330W AC Adapter"
                        )
                    ):
                        break

                    if (
                        next_key == "尺寸"
                        and line.startswith(
                            "357 x 254"
                        )
                    ):
                        break

                    if (
                        next_key == "重量"
                        and line.startswith(
                            "~2.5 kg"
                        )
                    ):
                        break

                    if (
                        next_key == "顏色"
                        and line == "Dark Tide"
                    ):
                        break

                    values.append(line)
                    current_index += 1

            specs[key] = values

        all_specs[variant] = specs

    return all_specs


def parse_specs(html):
    """
    統一的規格解析入口。

    1. 優先使用 Nuxt 結構化資料
    2. 找不到時退回可見文字解析
    """

    nuxt_specs = parse_nuxt_specs(html)

    if nuxt_specs is not None:
        return nuxt_specs

    return parse_visible_specs(html)


def save_specs(
    all_specs,
    output_path=OUTPUT_PATH,
):
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_specs,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():

    with open(
        HTML_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        html = file.read()

    all_specs = parse_specs(html)

    save_specs(
        all_specs,
        OUTPUT_PATH,
    )

    print("規格解析完成")
    print(
        f"已輸出到：{OUTPUT_PATH}"
    )

    for variant, specs in all_specs.items():

        print()
        print("=" * 60)
        print(variant)
        print("=" * 60)

        print("GPU:")
        for value in specs.get(
            "顯示晶片",
            [],
        ):
            print("-", value)

        print("RAM:")
        for value in specs.get(
            "記憶體",
            [],
        ):
            print("-", value)

        print("Battery:")
        for value in specs.get(
            "電池",
            [],
        ):
            print("-", value)


if __name__ == "__main__":
    main()