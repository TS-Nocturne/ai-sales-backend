"""Built-in technical reference for public specs (IP, Bluetooth, etc.).

Used as an offline fallback when vector search misses but the customer asks
a technical question the agent should still answer authoritatively.
"""

from __future__ import annotations

import re

# Each entry: (topic_id, title, trigger patterns, body text for tool output)
_TECHNICAL_ENTRIES: tuple[tuple[str, str, tuple[re.Pattern[str], ...], str], ...] = (
    (
        "ipx4",
        "มาตรฐาน IPX4 — กันละอองน้ำ",
        (
            re.compile(r"ipx\s*4\b", re.I),
            re.compile(r"ip\s*54", re.I),
        ),
        (
            "IPX4 คือมาตรฐานทดสอบ 'การกันละอองน้ำ' (splash resistant) "
            "เปรียบเทียบให้เห็นภาพ: ใส่ออกกำลังกายแล้วเหงื่อไหล หรือเดินกลางแจ้งแล้วเจอฝนโปรยๆ "
            "อุปกรณ์ระดับ IPX4 เอาอยู่สบายๆ "
            "แต่ห้ามใส่ลงไปว่ายน้ำหรือจุ่มน้ำโดยตรง เพราะเกินมาตรฐานที่รองรับ"
        ),
    ),
    (
        "ipx7",
        "มาตรฐาน IPX7 — จุ่มน้ำชั่วคราว",
        (re.compile(r"ipx\s*7\b", re.I),),
        (
            "IPX7 รองรับการจุ่มในน้ำนิ่งลึกประมาณ 1 เมตร เป็นเวลา 30 นาที (ตามเงื่อนไขทดสอบมาตรฐาน) "
            "เปรียบเทียบ: ตกลงสระตื้นๆ หรือโดนน้ำท่วมข้อมือชั่วคราว — มักรอด "
            "แต่ไม่ได้ออกแบบให้ว่ายน้ำเป็นเวลานานหรือใช้ในน้ำเค็ม"
        ),
    ),
    (
        "ip68",
        "มาตรฐาน IP68 — กันฝุ่นและกันน้ำเข้มข้น",
        (re.compile(r"ip\s*68\b", re.I), re.compile(r"ipx\s*8\b", re.I)),
        (
            "IP68 กันฝุ่นเข้าตัวเครื่องได้จริงจัง และกันน้ำได้ลึก/นานกว่า IPX7 "
            "ตามสเปกผู้ผลิต (เช่น 1.5–2 เมตร 30 นาที) "
            "เหมาะกับการใช้กลางฝนหนักหรือสภาพแวดล้อมที่มีฝุ่น/เหงื่อเยอะ "
            "แต่ยังไม่ควรดำน้ำลึกหรือแช่น้ำเค็มโดยไม่ตรวจคู่มือรุ่นนั้นๆ"
        ),
    ),
    (
        "ip_general",
        "มาตรฐาน IP Rating — ภาพรวม",
        (
            re.compile(r"\bipx?\s*\d", re.I),
            re.compile(r"ip\s*rating", re.I),
            re.compile(r"ingress\s*protection", re.I),
            re.compile(r"มาตรฐาน\s*ip", re.I),
            re.compile(r"ค่า\s*ip", re.I),
            re.compile(r"กันน้ำ", re.I),
            re.compile(r"water\s*proof", re.I),
            re.compile(r"water\s*resist", re.I),
            re.compile(r"กัน(?:ละออง|ฝุ่น)", re.I),
        ),
        (
            "IP Rating บอกว่าอุปกรณ์ทนน้ำ/ฝุ่นได้แค่ไหน: "
            "IPX4 = กันละอองน้ำ/เหงื่อ ใส่ออกกำลังกายได้ "
            "IPX7 = จุ่มน้ำชั่วคราวได้ "
            "IP68 = กันฝุ่น+กันน้ำเข้มข้น "
            "เลขยิ่งสูงไม่ได้หมายความว่าใส่ลงน้ำได้ทุกสถานการณ์ — ควรดูสเปกผู้ผลิตและอธิบายการใช้งานจริงให้ลูกค้า"
        ),
    ),
    (
        "bluetooth",
        "สเปก Bluetooth พื้นฐาน",
        (
            re.compile(r"bluetooth", re.I),
            re.compile(r"บลูทูธ", re.I),
            re.compile(r"\bbt\s*5", re.I),
            re.compile(r"\bcodec\b", re.I),
            re.compile(r"\baac\b", re.I),
            re.compile(r"aptx", re.I),
            re.compile(r"\bldac\b", re.I),
            re.compile(r"latency|ความหน่วง", re.I),
        ),
        (
            "Bluetooth 5.x มักมีระยะไกลขึ้นและเสถียรกว่ารุ่นเก่า "
            "Codec: SBC (มาตรฐาน), AAC (เหมาะ iPhone), aptX/LDAC (คุณภาพสูงบน Android ถ้าทั้งคู่รองรับ) "
            "Low latency / Gaming mode ลดความหน่วง เหมาะดูหนังและเล่นเกม"
        ),
    ),
)

# Order matters: specific IP levels before general IP topic.
_MATCH_ORDER = ("ipx4", "ipx7", "ip68", "ip_general", "bluetooth")


def looks_like_technical_query(text: str) -> bool:
    """True when the message is likely a technical/educational question."""
    return match_technical_reference(text) is not None


def match_technical_reference(query: str) -> dict | None:
    """Return a knowledge-shaped hit for built-in technical content, or None."""
    raw = (query or "").strip()
    if not raw:
        return None

    by_id = {entry[0]: entry for entry in _TECHNICAL_ENTRIES}
    for topic_id in _MATCH_ORDER:
        entry = by_id.get(topic_id)
        if not entry:
            continue
        _tid, title, patterns, body = entry
        if any(p.search(raw) for p in patterns):
            return {
                "source_type": "knowledge",
                "title": title,
                "text": body,
                "score": 1.0,
            }
    return None


def format_technical_reference(query: str) -> str | None:
    """Formatted tool output block for a technical reference hit."""
    hit = match_technical_reference(query)
    if not hit:
        return None
    return (
        f"[Technical Reference] Found 1 result(s) matching '{query}':\n\n"
        f"[Knowledge] {hit['title']}\n"
        f"  Relevance: 100%\n"
        f"  Content: {hit['text']}\n\n"
        "คำสั่ง: อธิบายให้ลูกค้าเห็นภาพด้วย analogy ชีวิตประจำวัน "
        "ห้ามตอบว่าไม่มีข้อมูล — ถ้ามีสินค้าที่กำลังคุยอยู่ให้เชื่อมกลับและชวนตัดสินใจ"
    )
