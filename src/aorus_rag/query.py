"""Conservative query normalization and scope checks for this specification corpus."""
import re
import unicodedata
from difflib import get_close_matches

VARIANTS = ('BZH', 'BYH', 'BXH')
ALIASES = {
    '作業系統': ['作業系統', 'os', 'operating system', 'windows'],
    '中央處理器': ['中央處理器', '處理器', 'cpu', 'processor', 'cores', '核心'],
    '顯示晶片': ['顯示晶片', '顯示卡', '顯卡', 'gpu', 'graphics', 'vram', '顯存'],
    '顯示器': ['顯示器', '螢幕', 'screen', 'display', 'monitor', '解析度', '更新率', '刷新率', 'refresh rate', 'resolution', 'oled'],
    '記憶體': ['記憶體', 'ram', 'memory', 'so-dimm'],
    '儲存裝置': ['儲存裝置', '儲存', '硬碟', 'ssd', 'storage', 'm.2'],
    '鍵盤種類': ['鍵盤', 'keyboard'],
    '連接埠': ['連接埠', '接口', 'port', 'ports', 'usb', 'hdmi', 'thunderbolt', 'microsd'],
    '音效': ['音效', '喇叭', 'speaker', 'speakers', 'audio', 'sound'],
    '通訊': ['通訊', 'wifi', 'wi-fi', 'bluetooth', '藍牙', '網路', 'network', 'lan'],
    '視訊鏡頭': ['視訊鏡頭', '攝影機', '鏡頭', 'webcam', 'camera', 'windows hello'],
    '安全裝置': ['安全裝置', '安全', 'security', 'tpm'],
    '電池': ['電池', 'battery'], '變壓器': ['變壓器', '充電器', 'adapter', 'charger'],
    '尺寸': ['尺寸', 'dimension', 'dimensions', 'size', '厚度'],
    '重量': ['重量', '多重', 'weight', 'weigh', 'kg'],
    '顏色': ['顏色', 'color', 'colour'],
}
LABELS = dict(zip(ALIASES, ['OS', 'CPU', 'GPU', 'Display', 'Memory', 'Storage', 'Keyboard', 'Ports', 'Audio', 'Connectivity', 'Webcam', 'Security', 'Battery', 'Power Adapter', 'Dimensions', 'Weight', 'Color']))
TYPO = {'記意體': '記憶體', '記億體': '記憶體', '記憶踢': '記憶體', '顯事卡': '顯示卡', '顯示喀': '顯示卡', '處理氣': '處理器', '螢暮': '螢幕', '電吃': '電池', '蓝牙': '藍牙', '内存': '記憶體', '显卡': '顯示卡', '屏幕': '螢幕', '处理器': '處理器', '电池': '電池'}


def has(text, term):
    pattern = re.escape(term)
    if term.isascii():
        pattern = rf'(?<![a-z0-9]){pattern}(?![a-z0-9])'
    return bool(re.search(pattern, text, re.I))


def normalize(query):
    query = unicodedata.normalize('NFKC', query).strip()
    for wrong, correct in TYPO.items():
        query = query.replace(wrong, correct)
    vocabulary = [a for values in ALIASES.values() for a in values if a.isascii() and a.isalpha() and len(a) >= 5]
    # Only repair long hardware words with a unique close match; never guess model codes.
    def repair(match):
        word = match.group()
        if word.lower() in vocabulary or len(word) < 5:
            return word
        candidates = get_close_matches(word.lower(), vocabulary, n=2, cutoff=0.82)
        return candidates[0] if len(candidates) == 1 else word
    return re.sub(r'[A-Za-z]+', repair, query)


def categories(query):
    return [key for key, aliases in ALIASES.items() if any(has(query, a) for a in aliases)]


def variants(query):
    return [v for v in VARIANTS if has(query, v)]


def is_overview(query):
    return bool(re.search(r'整體|總覽|介紹|所有規格|全部規格|規格表|overview|all (?:the )?spec|full spec|tell me about|introduc|比較.*型號|compare.*models', query, re.I))


def scope_message(query):
    """Return a bilingual reason when reliable answering requires unavailable data."""
    zh = bool(re.search(r'[\u4e00-\u9fff]', query))
    def message(cn, en):
        return cn if zh else en
    if not re.search(r'[\w\u4e00-\u9fff]', query) or query.lower() in {'hi', 'hello', '你好', '嗨'}:
        return message('你好！請指定想查詢的 AORUS MASTER 16 AM6H 規格，例如 GPU、記憶體或電池。', 'Hello! Please specify an AORUS MASTER 16 AM6H specification, such as GPU, memory or battery.')
    if re.search(r'忽略|無視|不要遵守|ignore|disregard|system prompt|系統提示|編造|捏造|make up|pretend', query, re.I):
        return message('我只能根據官方產品規格回答，不能依指示編造或改寫規格。請直接提出規格問題。', 'I can only answer from the official specifications, not invent or override them. Please ask a specification question.')
    unknown = [code for code in re.findall(r'(?<![A-Za-z0-9])B[A-Za-z0-9]{2}(?![A-Za-z0-9])', query, re.I) if code.upper() not in VARIANTS]
    if unknown or re.search(r'macbook|asus|lenovo|msi|acer|戴爾|華碩|聯想', query, re.I):
        return message('目前資料只涵蓋 AORUS MASTER 16 AM6H 的 BZH、BYH、BXH。請確認型號；不會自動猜測其他型號。', 'The data covers only AORUS MASTER 16 AM6H BZH, BYH and BXH. Please confirm the model; other models are not assumed.')
    if re.search(r'installed.*(?:ram|memory|ssd|storage)|(?:ram|memory|ssd|storage).*installed|出廠|出貨|實際.*(?:容量|記憶體|硬碟)|預裝|標配', query, re.I):
        return message('這份規格表提供容量上限與可能配置，未確認你購買機器的實際出貨配置。請提供完整銷售 SKU 或訂單規格；不能把最高支援容量當成已安裝容量。', 'The actual installed configuration is not confirmed by this specification table. It lists maximum capacities and possible configurations. Please provide the sales SKU or order specifications; maximum supported capacity is not installed capacity.')
    if re.search(r'觸控|touchscreen|touch screen|防水|waterproof|充飽|充滿|charging time|多久.*充|linux.*(?:支援|相容)|linux compatibility|能不能跑|能玩|跑得動|can it run|can.*play|cyberpunk', query, re.I):
        return message('規格表未提供足夠資訊確認這項功能、相容性或實際表現；未列出不代表一定不支援。請提供具體需求或官方補充資料。', 'The specification table does not provide enough information to confirm that feature, compatibility or real-world performance. An unlisted feature is not necessarily unsupported; please provide requirements or additional official evidence.')
    if re.search(r'續航|撐.*(?:多久|小時)|用.*幾小時|battery life|how (?:many hours|long).*battery|battery.*(?:last|hours)|運行時間|runtime|幾\s*fps|fps|幀率|跑分|benchmark|售價|價格|多少錢|price|cost|保固|warranty|庫存|stock|上市|release date|溫度|temperature|噪音|noise|散熱效果|推薦|值得買|適合|recommend|worth|suitable|遊戲.*效能|gaming performance', query, re.I):
        return message('官方規格表未提供這項實測、價格或使用情境資訊，無法據此準確判定。請提供相關測試或銷售資料；電池容量不等於續航時間，硬體規格也不等於遊戲 FPS。', 'The specification table does not provide the requested measurements, pricing or usage evidence. More test or sales data is needed; battery capacity does not establish battery life, and hardware specifications do not establish game FPS.')
    if re.search(r'天氣|weather|股票|stock price|食譜|recipe|笑話|joke|總統|president|寫.*(?:程式|詩)|write.*(?:code|poem)|\b[0-9]+\s*[+*/]\s*[0-9]+', query, re.I):
        return message('這部分不在產品規格資料範圍內；我可以協助查詢 AORUS MASTER 16 AM6H 的硬體規格。', 'That part is outside the product specification data. I can help with AORUS MASTER 16 AM6H hardware specifications.')
    return None
