"""
The three FitFindr tools.

Each one is a standalone function you can call and test on its own, before any
of them are wired into the loop. Build and test them one at a time — three
untested tools joined by a loop is one problem that looks like six, because you
can't tell which layer is lying to you.

    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)             → str
    create_fit_card(outfit, new_item)              → str

All three are stubs right now. They run and they do nothing — that's the
starting position and it's deliberate.

⚠️ Before you write any of them, fill in the **Tool Inventory** section of your
README (Milestone 2). Four lines per tool: what it does, each input with its
type, exactly what it returns, and what it returns when it has nothing to give.
That last line is what your loop branches on. "Returns a list" earns nothing —
the description has to say what is *in* the list.
"""

import config  # noqa: F401 — you'll use this in search_listings
from generate import generate
from utils.data_loader import load_listings


# ── Tool 1: search_listings ───────────────────────────────────────────────────
_STOPWORDS = {
    "a", "an", "and", "the", "for", "with", "under", "in", "of", "or", "to",
    "my", "me", "i", "looking", "want", "need", "some", "something", "any",
    "size", "please", "find", "get", "that", "this", "is", "it",
}


def _keywords(text: str) -> set[str]:
    """Lowercase words worth matching on, stopwords removed."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _size_tokens(size: str) -> set[str]:
    """
    The set of sizes one size string actually stands for.

    "S/M"              → {"S", "M"}
    "XL (oversized)"   → {"XL"}
    "W30 L30"          → {"W30 L30"}
    "US 8.5"           → {"US 8.5"}
    """
    cleaned = re.sub(r"\([^)]*\)", " ", size or "")        # drop parentheticals
    parts = [p.strip().upper() for p in cleaned.split("/")]
    return {p for p in parts if p}


def _size_matches(wanted: str, listing_size: str) -> bool:
    """
    Does a listing's size satisfy a requested size?

    ⚠️ This is the trap the starter's docstring warns about, and it is worth
    reading before you copy anything here. The obvious implementation is

        wanted.lower() in listing_size.lower()

    and it is wrong in both directions on this data:

        "s"  in "us 9"   → True   (a shoe returned for someone wanting a small)
        "l"  in "xl"     → True   (an XL returned for someone wanting a large)
        "m"  in "medium" → True   (accidentally right, for the wrong reason)

    So the comparison is between whole size *tokens*, not substrings. A listing
    matches when any of the sizes it stands for equals the one asked for.

    One deliberate exception: "One Size" matches every request. That is a
    judgment call rather than an obvious truth — it is in the Tool Inventory in
    the README because it is the kind of decision a reader would otherwise have
    to guess at.
    """
    if not wanted:
        return True

    listing_tokens = _size_tokens(listing_size)
    if any(token.startswith("ONE SIZE") for token in listing_tokens):
        return True

    return bool(_size_tokens(wanted) & listing_tokens)

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the listings data for items matching a description, and optionally a
    size and a price ceiling.

    This is the tool that doesn't call the model, which makes it the easiest one
    to test and the one to move onto MCP in unit 4.

    Args:
        description: keywords describing what the user wants
                     (e.g. "vintage graphic tee").
        size:        a size string to filter by, or None to skip size filtering.
                     Match case-insensitively — "M" should match "S/M".

                     ⚠️ Read the sizes in the data before you reach for a plain
                     substring test. `"s" in "us 9"` is True, and so is
                     `"l" in "xl"`. A filter that returns shoes when someone
                     asked for a small top reads like a broken search, and it
                     will quietly cost you in unit 4 when you test criterion 1.
                     What counts as a size match is part of your spec — decide
                     it and write it into your Tool Inventory.
        max_price:   maximum price, inclusive, or None to skip price filtering.

    Returns:
        A list of matching listing dicts, best match first.
        **Returns an empty list when nothing matches — an empty list, not None,
        and not an exception.** Your loop branches on this.

    Each listing dict has these fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand (str or None), platform

    Note that `brand` is None for most listings. That is deliberate and
    realistic — thrift listings often have no brand. If something you write
    assumes a brand is always there, you will find out in unit 4.

    TODO:
        1. Load every listing with load_listings().
        2. Filter by max_price and by size, when each is provided.
        3. Score what's left by keyword overlap with `description`.
        4. Drop anything scoring zero.
        5. Sort by score, highest first, and return the listing dicts —
           at most config.SEARCH_RESULT_LIMIT of them.

    Test it from a terminal before you move on:
        python -c "from tools import search_listings; print(search_listings('graphic tee', max_price=30))"
    """
    wanted = _keywords(description)

    scored: list[tuple[int, float, dict]] = []
    for listing in load_listings():
        if max_price is not None and listing["price"] > max_price:
            continue
        if size and not _size_matches(size, listing.get("size", "")):
            continue

        # Everything a keyword could reasonably match against. `brand` is None
        # on 32 of the 40 listings, so it is coerced rather than assumed.
        haystack = " ".join(
            [
                listing.get("title", ""),
                listing.get("description", ""),
                listing.get("category", ""),
                listing.get("brand") or "",
                " ".join(listing.get("style_tags", [])),
                " ".join(listing.get("colors", [])),
            ]
        )
        score = len(wanted & _keywords(haystack))
        if score:
            # Cheaper first among equal matches — a tie-break the user benefits
            # from, and it makes the result order deterministic, which unit 4's
            # state criterion depends on.
            scored.append((score, listing["price"], listing))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [listing for _, _, listing in scored[: config.SEARCH_RESULT_LIMIT]]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────
_OUTFIT_SYSTEM = (
    "You style thrifted clothing. Be concrete and brief. Name real garments, "
    "never colours alone. No preamble, no sign-off, no markdown headings."
)


def _describe_item(item: dict) -> str:
    """One line describing a listing, for a prompt."""
    return (
        f"{item.get('title')} — {item.get('category')}, "
        f"size {item.get('size')}, {item.get('condition')} condition, "
        f"colours: {', '.join(item.get('colors') or []) or 'unspecified'}, "
        f"style: {', '.join(item.get('style_tags') or []) or 'unspecified'}, "
        f"${item.get('price')} on {item.get('platform')}"
    )
    
    
def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest one or two outfits.

    This one calls the model, through `generate()`. You don't need to think
    about rate limits — the adapter handles pacing for you.

    Args:
        new_item: a listing dict — the item the user is considering.
        wardrobe: a wardrobe dict with an 'items' key holding a list of items.
                  **It may be empty.** Handle that.

    Returns:
        A non-empty string with outfit suggestions.
        With an empty wardrobe, return general styling advice rather than
        raising or returning "". Unit 4 has you trigger the empty wardrobe on
        purpose, so decide now what it should do.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If it is, ask the model for general styling ideas for this item.
        3. If it isn't, format the wardrobe items into the prompt and ask for
           specific combinations naming pieces the user already owns.
        4. Return the model's response.

    Test it from a terminal before you move on:
        python -c "from tools import suggest_outfit; from utils.data_loader import get_example_wardrobe, load_listings; print(suggest_outfit(load_listings()[0], get_example_wardrobe()))"
    """
    items = (wardrobe or {}).get("items") or []
    item_line = _describe_item(new_item)

    if not items:
        prompt = (
            f"Someone is thinking about buying this second-hand item:\n"
            f"  {item_line}\n\n"
            f"They have not saved a wardrobe, so you do not know what they own.\n"
            f"Suggest two outfits built around this item, describing each piece "
            f"generically ('straight-leg dark jeans', not a brand).\n"
            f"Open by saying these are general ideas because no wardrobe is "
            f"saved yet."
        )
    else:
        owned = "\n".join(
            f"  - {i.get('name')} ({i.get('category')}; "
            f"{', '.join(i.get('colors') or []) or 'colour unspecified'})"
            for i in items
        )
        prompt = (
            f"Someone is thinking about buying this second-hand item:\n"
            f"  {item_line}\n\n"
            f"They already own:\n{owned}\n\n"
            f"Suggest two outfits pairing the new item with pieces they already "
            f"own. Name the owned pieces exactly as they are written above. "
            f"Do not invent pieces they do not have."
        )

    return generate(prompt, system=_OUTFIT_SYSTEM)


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────
_CARD_SYSTEM = (
    "You write short captions for second-hand fashion finds, in the voice of "
    "the person who found it. Two to four sentences. No hashtag walls, no "
    "markdown, no headings."
)

NO_OUTFIT_MESSAGE = (
    "No fit card — create_fit_card was called with no outfit suggestion, so "
    "there was nothing to write about. Check that suggest_outfit returned "
    "something before this step."
)

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Write a short caption someone would actually post about the find.

    This calls the model too.

    Args:
        outfit:   the outfit suggestion string from suggest_outfit().
        new_item: the listing dict for the item.

    Returns:
        A two-to-four sentence caption.
        If `outfit` is empty or whitespace, return a descriptive message rather
        than raising.

    The caption should read like a real post rather than a product description,
    mention the item and its price and platform once each, and be specific about
    the vibe.

    It should also come out **differently for different inputs**. If you run
    this three times on the same item and get three word-for-word identical
    strings, it's one of two things, and both are near the top of `config.py`:

        • CACHE_ENABLED — the adapter handed back an answer it already had
        • TEMPERATURE   — at 0.0 the model gives the same words every time

    TODO:
        1. Guard against an empty or whitespace-only `outfit`.
        2. Build a prompt with the item details and the outfit.
        3. Call generate() and return the response.

    Test it from a terminal before you move on:
        python -c "from tools import create_fit_card; from utils.data_loader import load_listings; print(create_fit_card('jeans and white sneakers', load_listings()[0]))"
    """
    if not (outfit or "").strip():
        return NO_OUTFIT_MESSAGE
