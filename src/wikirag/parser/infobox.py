import re
from typing import Any, Dict, List, Optional
import mwparserfromhell
from wikirag.parser.models import InfoboxData


def clean_wikitext_value(val: str) -> str:
    """Strips common MediaWiki syntax and helper templates from field values.
    
    Handles:
    - {{S|...}} -> notes/context
    - {{Ref|...}} -> remove reference tags
    - [[Target|Display]] -> Display
    - [[Target]] -> Target
    - <br>, <s>, </s>, <gallery>...</gallery>
    - HTML entities and comments
    """
    if not val:
        return ""

    # Remove HTML comments
    val = re.sub(r"<!--.*?-->", "", val, flags=re.DOTALL)

    # Remove galleries
    val = re.sub(r"<gallery>.*?</gallery>", "", val, flags=re.DOTALL)

    # Remove <ref> tags and contents
    val = re.sub(r"<ref[^>]*>.*?</ref>", "", val, flags=re.DOTALL)
    val = re.sub(r"<ref[^>]*/>", "", val)

    # Replace <br>, <br/>, <br /> with newline
    val = re.sub(r"<br\s*/?>", "\n", val, flags=re.IGNORECASE)

    # Parse with mwparserfromhell to strip template markers
    try:
        parsed = mwparserfromhell.parse(val)
        # Custom handling for {{S|note}} -> (note)
        for t in parsed.filter_templates():
            t_name = str(t.name).strip()
            if t_name == "S" and t.params:
                t_val = str(t.params[0].value).strip()
                parsed.replace(t, f" ({t_val})")
            elif t_name == "Ref":
                parsed.remove(t)
            elif t_name == "Nihongo" and t.params:
                # {{Nihongo|English|Kanji|Romaji}}
                p0 = str(t.params[0].value).strip()
                parsed.replace(t, p0)
            elif t_name == "W" and t.params:
                # {{W|Person}} -> Person
                parsed.replace(t, str(t.params[0].value).strip())
            else:
                parsed.remove(t)

        # Convert wikilinks [[Target|Label]] -> Label
        for link in parsed.filter_wikilinks():
            text = str(link.text or link.title).strip()
            parsed.replace(link, text)

        val = parsed.strip_code()
    except Exception:
        # Fallback to regex if AST parser encounters malformed wikitext
        val = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", val)
        val = re.sub(r"\{\{[^}]+\}\}", "", val)

    # Clean up whitespace and remaining HTML tags
    val = re.sub(r"<[^>]+>", "", val)
    val = re.sub(r"[ \t]+", " ", val).strip()
    return val


def split_multivalue_field(val: str) -> List[str]:
    """Splits a cleaned field containing multiple items separated by newlines, commas, or bullets."""
    lines = [line.strip("•-* ") for line in val.split("\n") if line.strip("•-* ")]
    res = []
    for line in lines:
        if "," in line and not ("(" in line and ")" in line):
            res.extend([item.strip() for item in line.split(",") if item.strip()])
        else:
            res.append(line)
    return [r for r in res if r]


def extract_infobox(wikicode: mwparserfromhell.wikicode.Wikicode) -> Optional[InfoboxData]:
    """Finds and extracts any Infobox template (Character, Location, etc.) into structured InfoboxData."""
    templates = wikicode.filter_templates()
    infobox_template = None

    for t in templates:
        name = str(t.name).strip().lower()
        if "infobox" in name:
            infobox_template = t
            break

    if not infobox_template:
        return None

    raw_fields: Dict[str, str] = {}
    for p in infobox_template.params:
        key = str(p.name).strip()
        raw_val = str(p.value).strip()
        cleaned_val = clean_wikitext_value(raw_val)
        if cleaned_val:
            raw_fields[key] = cleaned_val

    # Construct typed fields with common Fandom aliases
    def get_field(*keys: str) -> Optional[str]:
        for k in keys:
            if k in raw_fields:
                return raw_fields[k]
        return None

    def get_list(*keys: str) -> List[str]:
        for k in keys:
            if k in raw_fields:
                return split_multivalue_field(raw_fields[k])
        return []

    return InfoboxData(
        name=get_field("name", "title", "kanji_name"),
        kanji=get_field("kanji"),
        romaji=get_field("romaji"),
        aliases=get_list("alias", "aliases", "fnames"),
        species=get_field("species", "race", "evolution"),
        gender=get_field("gender"),
        age=get_field("age"),
        status=get_field("status"),
        country=get_field("country", "base", "origin"),
        affiliation=get_list("affiliation", "faffiliation"),
        occupation=get_list("occupations", "occupation", "class"),
        titles=get_list("titles", "ftitles"),
        family=get_list("family", "relatives"),
        rank=get_field("rank", "drank", "arank"),
        first_appearance=get_field("light", "manga", "anime", "web", "first_appearance"),
        image_url=get_field("image", "image1"),
        raw_fields=raw_fields,
    )
