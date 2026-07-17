"""Unicode script detection helpers for mixed-script PII routing.

The helpers in this module are intentionally lightweight and stdlib-only. They
use explicit Unicode block ranges plus :mod:`unicodedata` character categories
to identify dominant scripts and preserve exact offsets while segmenting text
into script-oriented runs.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

UNKNOWN_SCRIPT = "Unknown"


class ChineseScriptVariant(str, Enum):
    """Estimated Chinese character variant used in a text."""

    SIMPLIFIED = "simplified"
    TRADITIONAL = "traditional"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ChineseScriptEstimate:
    """Ratio-based Simplified/Traditional estimate for synthetic text."""

    variant: ChineseScriptVariant
    simplified_count: int
    traditional_count: int
    simplified_ratio: float
    traditional_ratio: float

    @property
    def script(self) -> ChineseScriptVariant:
        """Alias for the estimated predominant variant."""

        return self.variant

    @property
    def mixed(self) -> bool:
        """Return whether both Simplified and Traditional evidence appears."""

        return self.simplified_count > 0 and self.traditional_count > 0


SUPPORTED_SCRIPTS = (
    "Latin",
    "Arabic",
    "Han",
    "Hiragana/Katakana",
    "Hangul",
    "Cyrillic",
    "Devanagari",
    "Telugu",
    "Greek",
    "Hebrew",
    "Thai",
)

SCRIPT_LANGUAGE_HINTS: dict[str, tuple[str, ...]] = {
    "Latin": ("en", "fr", "de", "it", "es", "nl", "pt", "tr"),
    "Arabic": ("ar",),
    "Han": ("ja",),
    "Hiragana/Katakana": ("ja",),
    "Hangul": ("ko",),
    "Cyrillic": ("en",),
    "Devanagari": ("hi",),
    "Telugu": ("te",),
    "Greek": ("en",),
    "Hebrew": ("en",),
    "Thai": ("en",),
    UNKNOWN_SCRIPT: ("en",),
}

ZERO_WIDTH_CHARS = frozenset(
    {
        "\u200b",  # zero width space
        "\u200c",  # zero width non-joiner
        "\u200d",  # zero width joiner
        "\u2060",  # word joiner
        "\ufeff",  # zero width no-break space
    }
)

_CONFUSABLE_FOLD: dict[str, str] = {
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a7": "X",
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b7": "n",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03bc": "u",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03c7": "x",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041c": "M",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0422": "T",
    "\u0425": "X",
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0445": "x",
    "\u0456": "i",
}

# Unicode assigns both Simplified and Traditional Han characters to the same
# script blocks. These curated, common one-sided variants provide deterministic
# evidence without bundling a language model or copying OpenCC dictionaries.
_SIMPLIFIED_VARIANT_RAW = frozenset(
    "头药医门发后里网软台历叶万与东丝两严个临为乐书买乱云产亲仅从仓们"
    "众会体余传伤伦儿兰关兴养写军农冲决况冻净减几凤击刘创别动务华"
    "单卫厂厅压厌县双变号听员园圆场声处复备够夹夺奖妇妈孙学宁宝实"
    "审宽对寻导将层岁岛岭帐帮广庄庆库应开张归当录忆怀总态恋恶惊惯"
    "愿护报担拟拥拦拨择挂挤挥损换据摆敌数无旧时显晓术机杀杂权条来"
    "杨极构枪柜树样档桥梦检欢欧残气汉汤沟泪泻泽洁浅测济浓涛涡润涨"
    "湿滚满滤灭灯灵点炼热爱爷牵状犹独狮猫猪环现电画疗疮疯痒瘫盘盐"
    "监盖盗矿码砖确础礼祷离种积称稳穷窍竞笔笼筑签简粮紧红纤约级纪"
    "纯线练组细织终经结给络绝统绢继续绿编缘缝罗罚翘职联聪肃肠肤肿"
    "胀胜胶脉脏脑脸艺节芜苹范茧荐荡荣莱莲获营萧萨蓝虑虚虫虽虾蚁蚕"
    "补装见观规视览觉触计订认讲许论设访证评诉诊词译试诗诚话该详语"
    "误说请诸读课调谈谢谣谱贝财责贤账货质购贵费贺资赌赏赔赖赞赢赵"
    "车转轮轻载轿较辅辆辈输辑边达迁过运还这进远连迟适选递逻邓郑酱"
    "释鉴钟钢钥钱铁铜铝银销锁锅锋锐错锡锣锦键锯镜长闪闭问闯闲间闷"
    "闸闹闻阁阅阔队阳阴阵阶际陆陈险随隐难雾静韩页顶项顺须顾顿领颈"
    "频题颜额风飞饭饮馆马驱驶骑验鱼鲜鲸鸟鸡鸭鹅鹤鹰黄齐齿龄龙龟"
)
_TRADITIONAL_VARIANT_RAW = frozenset(
    "頭藥醫門發髮後裡裏網軟臺歷葉萬與東絲兩嚴個臨為樂書買亂雲產親"
    "僅從倉們眾會體餘傳傷倫兒蘭關興養寫軍農衝決況凍淨減幾鳳擊劉創"
    "別動務華單衛廠廳壓厭縣雙變號聽員園圓場聲處復備夠夾奪獎婦媽孫"
    "學寧寶實審寬對尋導將層歲島嶺帳幫廣莊慶庫應開張歸當錄憶懷總態"
    "戀惡驚慣願護報擔擬擁攔撥擇掛擠揮損換據擺敵數無舊時顯曉術機殺"
    "雜權條來楊極構槍櫃樹樣檔橋夢檢歡歐殘氣漢湯溝淚瀉澤潔淺測濟濃"
    "濤渦潤漲濕滾滿濾滅燈靈點煉熱愛爺牽狀猶獨獅貓豬環現電畫療瘡瘋"
    "癢癱盤鹽監蓋盜礦碼磚確礎禮禱離種積稱穩窮竅競筆籠築簽簡糧緊紅"
    "纖約級紀純線練組細織終經結給絡絕統絹繼續綠編緣縫羅罰翹職聯聰"
    "肅腸膚腫脹勝膠脈臟腦臉藝節蕪蘋範繭薦蕩榮萊蓮獲營蕭薩藍慮虛蟲"
    "雖蝦蟻蠶補裝見觀規視覽覺觸計訂認講許論設訪證評訴診詞譯試詩誠"
    "話該詳語誤說請諸讀課調談謝謠譜貝財責賢賬貨質購貴費賀資賭賞賠"
    "賴贊贏趙車轉輪輕載轎較輔輛輩輸輯邊達遷過運還這進遠連遲適選遞"
    "邏鄧鄭醬釋鑒鐘鋼鑰錢鐵銅鋁銀銷鎖鍋鋒銳錯錫鑼錦鍵鋸鏡長閃閉問"
    "闖閒間悶閘鬧聞閣閱闊隊陽陰陣階際陸陳險隨隱難霧靜韓頁頂項順須"
    "顧頓領頸頻題顏額風飛飯飲館馬驅駛騎驗魚鮮鯨鳥雞鴨鵝鶴鷹黃齊齒"
    "齡龍龜"
)
_SHARED_VARIANT_EVIDENCE = _SIMPLIFIED_VARIANT_RAW & _TRADITIONAL_VARIANT_RAW
SIMPLIFIED_VARIANT_CHARS = _SIMPLIFIED_VARIANT_RAW - _SHARED_VARIANT_EVIDENCE
TRADITIONAL_VARIANT_CHARS = _TRADITIONAL_VARIANT_RAW - _SHARED_VARIANT_EVIDENCE


@dataclass(frozen=True)
class DetectionNormalization:
    """Offset-preserving Unicode normalization for PII detection."""

    text: str
    original_length: int
    offset_starts: tuple[int, ...]
    offset_ends: tuple[int, ...]
    removed_zero_width: int = 0
    stripped_combining_marks: int = 0
    folded_confusables: int = 0
    folded_native_digits: int = 0
    scripts: tuple[str, ...] = ()
    mixed_script: bool = False
    chinese_variant_normalized: bool = False
    chinese_target_script: str | None = None
    opencc_available: bool | None = None

    @property
    def changed(self) -> bool:
        """Return whether the normalized text differs structurally."""
        return (
            self.removed_zero_width > 0
            or self.stripped_combining_marks > 0
            or self.folded_confusables > 0
            or self.folded_native_digits > 0
            or self.chinese_variant_normalized
        )

    def remap_span(self, start: int, end: int) -> tuple[int, int]:
        """Map normalized-text offsets back to original-text offsets."""
        safe_start = max(0, min(int(start), len(self.text)))
        safe_end = max(safe_start, min(int(end), len(self.text)))
        if not self.offset_starts:
            return 0, 0
        if safe_start >= len(self.offset_starts):
            original_start = self.original_length
        else:
            original_start = self.offset_starts[safe_start]
        if safe_end <= 0:
            original_end = original_start
        elif safe_end - 1 >= len(self.offset_ends):
            original_end = self.original_length
        else:
            original_end = self.offset_ends[safe_end - 1]
        return original_start, max(original_start, original_end)

    def to_metadata(self) -> dict[str, object]:
        """Return PHI-free normalization metadata."""
        return {
            "changed": self.changed,
            "chinese_target_script": self.chinese_target_script,
            "chinese_variant_normalized": self.chinese_variant_normalized,
            "folded_confusables": self.folded_confusables,
            "folded_native_digits": self.folded_native_digits,
            "mixed_script": self.mixed_script,
            "opencc_available": self.opencc_available,
            "removed_zero_width": self.removed_zero_width,
            "scripts": list(self.scripts),
            "stripped_combining_marks": self.stripped_combining_marks,
        }


_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    (
        "Latin",
        (
            (0x0041, 0x005A),
            (0x0061, 0x007A),
            (0x00C0, 0x00FF),
            (0x0100, 0x017F),
            (0x0180, 0x024F),
            (0x1E00, 0x1EFF),
            (0x2C60, 0x2C7F),
            (0xA720, 0xA7FF),
            (0xAB30, 0xAB6F),
            (0xFF21, 0xFF3A),
            (0xFF41, 0xFF5A),
        ),
    ),
    (
        "Arabic",
        (
            (0x0600, 0x06FF),
            (0x0750, 0x077F),
            (0x08A0, 0x08FF),
            (0xFB50, 0xFDFF),
            (0xFE70, 0xFEFF),
        ),
    ),
    (
        "Han",
        (
            (0x3400, 0x4DBF),
            (0x4E00, 0x9FFF),
            (0xF900, 0xFAFF),
            (0x20000, 0x2A6DF),
            (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F),
            (0x2B820, 0x2CEAF),
            (0x2CEB0, 0x2EBEF),
            (0x30000, 0x3134F),
            (0x31350, 0x323AF),
        ),
    ),
    (
        "Hiragana/Katakana",
        (
            (0x3040, 0x309F),
            (0x30A0, 0x30FF),
            (0x31F0, 0x31FF),
            (0x1B000, 0x1B16F),
            (0xFF65, 0xFF9F),
        ),
    ),
    (
        "Hangul",
        (
            (0x1100, 0x11FF),
            (0x3130, 0x318F),
            (0xA960, 0xA97F),
            (0xAC00, 0xD7AF),
            (0xD7B0, 0xD7FF),
        ),
    ),
    (
        "Cyrillic",
        (
            (0x0400, 0x04FF),
            (0x0500, 0x052F),
            (0x1C80, 0x1C8F),
            (0x2DE0, 0x2DFF),
            (0xA640, 0xA69F),
        ),
    ),
    (
        "Devanagari",
        (
            (0x0900, 0x097F),
            (0xA8E0, 0xA8FF),
            (0x11B00, 0x11B5F),
        ),
    ),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    (
        "Greek",
        (
            (0x0370, 0x03FF),
            (0x1F00, 0x1FFF),
        ),
    ),
    (
        "Hebrew",
        (
            (0x0590, 0x05FF),
            (0xFB1D, 0xFB4F),
        ),
    ),
    ("Thai", ((0x0E00, 0x0E7F),)),
)


def detect_script(text: str) -> str:
    """Return the dominant Unicode script in ``text``.

    Neutral characters such as whitespace, punctuation, symbols, and digits do
    not affect the decision. If no supported script-bearing code point is
    present, ``"Unknown"`` is returned.
    """

    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        first_seen.setdefault(script, index)

    if not counts:
        return UNKNOWN_SCRIPT

    return max(counts, key=lambda script: (counts[script], -first_seen[script]))


def detect_chinese_script(text: str) -> ChineseScriptEstimate:
    """Estimate the predominant Simplified/Traditional Chinese variant.

    Unicode blocks cannot distinguish the variants, so the estimate uses
    ratios over common characters that are exclusive to one form. Characters
    shared by both forms are ignored. A tie with evidence on both sides is
    reported as mixed; otherwise the larger evidence ratio is predominant.
    """

    simplified_count = sum(char in SIMPLIFIED_VARIANT_CHARS for char in text)
    traditional_count = sum(char in TRADITIONAL_VARIANT_CHARS for char in text)
    evidence_count = simplified_count + traditional_count
    if evidence_count == 0:
        variant = ChineseScriptVariant.UNKNOWN
        simplified_ratio = 0.0
        traditional_ratio = 0.0
    else:
        simplified_ratio = simplified_count / evidence_count
        traditional_ratio = traditional_count / evidence_count
        if simplified_count == traditional_count:
            variant = ChineseScriptVariant.MIXED
        elif simplified_count > traditional_count:
            variant = ChineseScriptVariant.SIMPLIFIED
        else:
            variant = ChineseScriptVariant.TRADITIONAL

    return ChineseScriptEstimate(
        variant=variant,
        simplified_count=simplified_count,
        traditional_count=traditional_count,
        simplified_ratio=simplified_ratio,
        traditional_ratio=traditional_ratio,
    )


def segment_by_script(text: str) -> Iterator[tuple[int, int, str]]:
    """Yield contiguous ``(start, end, script)`` runs covering ``text``.

    Neutral characters are assigned to the surrounding run: leading neutral
    characters attach to the first detected script, and later neutral characters
    attach to the preceding script. This keeps offsets exact while avoiding
    stand-alone whitespace or punctuation runs.
    """

    if not text:
        return

    run_start = 0
    current_script: str | None = None

    for index, char in enumerate(text):
        script = _script_for_char(char)
        if script is None:
            continue
        if current_script is None:
            current_script = script
            continue
        if script == current_script:
            continue

        yield run_start, index, current_script
        run_start = index
        current_script = script

    if current_script is None:
        yield 0, len(text), UNKNOWN_SCRIPT
        return

    yield run_start, len(text), current_script


def candidate_languages_for_script(script: str) -> tuple[str, ...]:
    """Return candidate language codes for a detected script."""

    return SCRIPT_LANGUAGE_HINTS.get(script, SCRIPT_LANGUAGE_HINTS[UNKNOWN_SCRIPT])


def normalize_for_pii_detection(
    text: str,
    *,
    width_convention: str = "cjk",
    chinese_target_script: str | None = None,
) -> DetectionNormalization:
    """Fold adversarial Unicode artifacts while preserving offset remapping.

    The defense strips zero-width controls and standalone combining marks, folds
    common Latin-lookalike Greek/Cyrillic/full-width characters, folds Indic
    decimal digits for ASCII validators, and records a script-consistency
    summary without storing source text. ``width_convention`` selects the
    CJK-safe width fold or strict per-character NFKC normalization.
    ``chinese_target_script`` optionally canonicalizes Han variants with OpenCC
    after the Unicode defenses and composes its alignment into the source map.
    """

    # Keep the reusable width-normalization API in ``processing`` while
    # composing its explicit source map with this existing detection defense.
    # The local import avoids making the lightweight script helpers import the
    # broader processing package during module initialization.
    from ..processing.text import fold_indic_digits
    from ..processing.zh_normalize import normalize_chinese_variants, normalize_width

    scripts = tuple(sorted(_script_counts(text)))
    width_normalization = normalize_width(text, convention=width_convention)
    digit_folding = fold_indic_digits(width_normalization.text)
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    removed_zero_width = 0
    stripped_combining_marks = 0
    normalized_by_source: list[list[str]] = [[] for _ in text]
    for char, (original_start, _original_end) in zip(
        width_normalization.text,
        width_normalization.char_origins,
    ):
        normalized_by_source[original_start].append(char)
    changed_source_indices = {
        index
        for index, (char, normalized_chars) in enumerate(
            zip(text, normalized_by_source)
        )
        if "".join(normalized_chars) != char
    }
    folded_native_digit_sources = {
        width_normalization.char_origins[index][0]
        for index, (width_char, folded_char) in enumerate(
            zip(width_normalization.text, digit_folding.text)
        )
        if width_char != folded_char
    }

    for index, char in enumerate(digit_folding.text):
        original_start, original_end = width_normalization.char_origins[index]
        if char in ZERO_WIDTH_CHARS:
            removed_zero_width += 1
            continue
        if unicodedata.category(char) == "Mn":
            stripped_combining_marks += 1
            continue

        replacement = _fold_confusable_char(char)
        if replacement != char:
            changed_source_indices.add(original_start)
        for replacement_char in replacement:
            output.append(replacement_char)
            starts.append(original_start)
            ends.append(original_end)

    normalized_text = "".join(output)
    chinese_variant_normalized = False
    opencc_available: bool | None = None
    if chinese_target_script is not None:
        conversion = normalize_chinese_variants(
            normalized_text,
            chinese_target_script,
        )
        converted_starts: list[int] = []
        converted_ends: list[int] = []
        for converted_start, converted_end in conversion.char_origins:
            if converted_start < converted_end:
                source_starts = starts[converted_start:converted_end]
                source_ends = ends[converted_start:converted_end]
                converted_starts.append(min(source_starts))
                converted_ends.append(max(source_ends))
            else:
                anchor = (
                    starts[converted_start]
                    if converted_start < len(starts)
                    else len(text)
                )
                converted_starts.append(anchor)
                converted_ends.append(anchor)
        normalized_text = conversion.text
        starts = converted_starts
        ends = converted_ends
        chinese_variant_normalized = conversion.changed
        opencc_available = conversion.opencc_available

    return DetectionNormalization(
        text=normalized_text,
        original_length=len(text),
        offset_starts=tuple(starts),
        offset_ends=tuple(ends),
        removed_zero_width=removed_zero_width,
        stripped_combining_marks=stripped_combining_marks,
        folded_confusables=len(changed_source_indices),
        folded_native_digits=len(folded_native_digit_sources),
        scripts=scripts,
        mixed_script=len(scripts) > 1,
        chinese_variant_normalized=chinese_variant_normalized,
        chinese_target_script=chinese_target_script,
        opencc_available=opencc_available,
    )


def _script_for_char(char: str) -> str | None:
    category = unicodedata.category(char)
    if category[0] not in {"L", "M"}:
        return None

    codepoint = ord(char)
    for script, ranges in _SCRIPT_RANGES:
        if any(start <= codepoint <= end for start, end in ranges):
            return script
    return None


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        script = _script_for_char(char)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
    return counts


def _fold_confusable_char(char: str) -> str:
    folded = _CONFUSABLE_FOLD.get(char)
    if folded is not None:
        return folded

    codepoint = ord(char)
    if 0xFF01 <= codepoint <= 0xFF5E:
        return chr(codepoint - 0xFEE0)

    return char


__all__ = [
    "ChineseScriptEstimate",
    "ChineseScriptVariant",
    "DetectionNormalization",
    "SCRIPT_LANGUAGE_HINTS",
    "SIMPLIFIED_VARIANT_CHARS",
    "SUPPORTED_SCRIPTS",
    "TRADITIONAL_VARIANT_CHARS",
    "UNKNOWN_SCRIPT",
    "ZERO_WIDTH_CHARS",
    "candidate_languages_for_script",
    "detect_chinese_script",
    "detect_script",
    "normalize_for_pii_detection",
    "segment_by_script",
]
