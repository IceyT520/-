"""纯 Python Markdown -> docx 转换(仅标准库, 不依赖 python-docx/lxml)。
本机 python-docx 的 lxml xpath 会崩溃, 故直接手工拼装 docx 的 XML。
支持: 标题/加粗/表格/列表/代码块/引用/分隔线。
用法: python tools/md_to_docx_pure.py <输入.md> <输出.docx>
"""

import re
import sys
import zipfile
from xml.sax.saxutils import escape

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
<w:rPr><w:rFonts w:ascii="Calibri" w:eastAsia="微软雅黑"/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:before="360" w:after="160"/><w:outlineLvl w:val="0"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="36"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:before="300" w:after="140"/><w:outlineLvl w:val="1"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:before="240" w:after="120"/><w:outlineLvl w:val="2"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading4"><w:name w:val="heading 4"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:before="200" w:after="100"/><w:outlineLvl w:val="3"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="23"/></w:rPr></w:style>
</w:styles>"""


def _runs(text: str) -> str:
    """行内 **bold** 与 `code` -> 带格式的 run XML。"""
    out = []
    for token in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**"):
            rpr, body = "<w:b/>", escape(token[2:-2])
        elif token.startswith("`") and token.endswith("`"):
            rpr, body = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:color w:val="B03030"/>', escape(token[1:-1])
        else:
            rpr, body = "", escape(token)
        out.append(
            f'<w:r><w:rPr><w:rFonts w:eastAsia="微软雅黑"/>{rpr}</w:rPr>'
            f'<w:t xml:space="preserve">{body}</w:t></w:r>'
        )
    return "".join(out)


def _p(text: str, style: str = "", indent: int = 0) -> str:
    ppr = ""
    inner = f'<w:pStyle w:val="{style}"/>' if style else ""
    if indent:
        inner += f'<w:ind w:left="{indent}"/>'
    if inner:
        ppr = f"<w:pPr>{inner}</w:pPr>"
    return f"<w:p>{ppr}{_runs(text)}</w:p>"


def _table(header: list[str], rows: list[list[str]]) -> str:
    n = len(header)
    grid = "".join(f'<w:gridCol w:w="{9000 // max(n,1)}"/>' for _ in range(n))

    def cell(text: str, bold: bool) -> str:
        return (
            '<w:tc><w:tcPr><w:tcW w:w="0" w:type="auto"/></w:tcPr>'
            + _p(f"**{text}**" if bold else text)
            + "</w:tc>"
        )

    xml = [
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        "<w:tblBorders>"
        + "".join(
            f'<w:{b} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
            for b in ("top", "left", "bottom", "right", "insideH", "insideV")
        )
        + "</w:tblBorders></w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
    ]
    xml.append("<w:tr>" + "".join(cell(h, True) for h in header) + "</w:tr>")
    for row in rows:
        xml.append(
            "<w:tr>" + "".join(cell(c, False) for c in (row + [""] * n)[:n]) + "</w:tr>"
        )
    xml.append("</w:tbl>")
    return "".join(xml)


def convert(md_path: str, docx_path: str):
    lines = open(md_path, encoding="utf-8").read().splitlines()
    body = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            for b in buf:
                body.append(_p(f"`{b}`", indent=360))
            continue
        if (
            line.strip().startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s\-|]+\|\s*$", lines[i + 1])
        ):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body.append(_table(header, rows))
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            body.append(_p(m.group(2), style=f"Heading{len(m.group(1))}"))
            i += 1
            continue
        if re.match(r"^\s*---+\s*$", line):
            body.append(_p("─" * 40))
            i += 1
            continue
        if line.strip().startswith(">"):
            body.append(_p(line.strip().lstrip(">").strip(), indent=360))
            i += 1
            continue
        m = re.match(r"^\s*([-*+]|\d+[.、)])\s+(.*)$", line)
        if m:
            marker = "• " if not re.match(r"\d", m.group(1)) else m.group(1) + " "
            body.append(_p(marker + m.group(2), indent=360))
            i += 1
            continue
        body.append(_p(line))
        i += 1

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>" + "".join(body) + "</w:body></w:document>"
    )
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _RELS)
        z.writestr("word/styles.xml", _STYLES)
        z.writestr("word/document.xml", document)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    print(f"已生成 {sys.argv[2]}")
