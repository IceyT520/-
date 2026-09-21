"""显示层文本美化: 把 ASCII 化学式渲染为上下标格式。

仅用于界面展示; 检索与存储层仍使用 ASCII 形式(Li3YCl6), 互不干扰。
转换规则:
- 元素符号后的数字 -> 下标: Li3YCl6 -> Li₃YCl₆, Li2.6In0.8Ta0.2Cl6 -> Li₂.₆In₀.₈Ta₀.₂Cl₆
- 科学计数 10-3 -> 10⁻³ (含 Unicode 负号 U+2212 的写法)
- 单位 cm-1/cm-2 -> cm⁻¹/cm⁻²
行首化学计量比(如 2LiF、1.6Li2O 中的前导数字)不转换, 与习惯写法一致。
"""

import re

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# 显式元素表(两字母在前保证贪婪匹配), 避免误伤 "IceyT520" 这类字符串
_ELEMENTS = (
    "Li|Na|Mg|Al|Si|Cl|Ar|Ca|Sc|Ti|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|As|Se|Br|Kr|Rb|Sr|"
    "Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|Sb|Te|Xe|Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|"
    "Dy|Ho|Er|Tm|Yb|Lu|Hf|Ta|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|He|Be|B|C|N|O|F|Ne|P|S|K|V|Y|I|W|H"
)
_ELEM_NUM = re.compile(rf"({_ELEMENTS})(\d+\.?\d*)")
_SCI = re.compile(r"10[\-−](\d+)")
_CM = re.compile(r"cm[\-−](\d+)")


def prettify(text: str) -> str:
    if not text:
        return text
    text = _ELEM_NUM.sub(lambda m: m.group(1) + m.group(2).translate(_SUB), text)
    text = _SCI.sub(lambda m: "10⁻" + m.group(1).translate(_SUP), text)
    text = _CM.sub(lambda m: "cm⁻" + m.group(1).translate(_SUP), text)
    return text
