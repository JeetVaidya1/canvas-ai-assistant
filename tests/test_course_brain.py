"""course_brain — synthesis, persistence, reads, and label matching (hermetic).

Supabase is swapped for FakeSupabase and the LLM for a canned structured_call,
following the established fakes_endpoints pattern.
"""
import pytest

import course_brain
from course_brain import Topic, match_mastery, match_mastery_rows, match_topic, slugify
from fakes_endpoints import FakeSupabase

pytestmark = pytest.mark.unit

COURSE = "cs101"


def _content_db(course_topics=None):
    """A course with two ingested docs and optional pre-existing topics."""
    return FakeSupabase({
        "files": [
            {"course_id": COURSE, "filename": "trees.pdf"},
            {"course_id": COURSE, "filename": "hashing.pdf"},
            {"course_id": "other", "filename": "other.pdf"},
        ],
        "embeddings": [
            {"course_id": COURSE, "doc_name": "trees.pdf", "chunk_id": 1,
             "content": "Binary search trees store ordered keys.", "page": 2, "slide": None},
            {"course_id": COURSE, "doc_name": "trees.pdf", "chunk_id": 2,
             "content": "Tree rotations keep AVL trees balanced.", "page": 9, "slide": None},
            {"course_id": COURSE, "doc_name": "hashing.pdf", "chunk_id": 1,
             "content": "Hash functions map keys to buckets.", "page": 1, "slide": None},
            {"course_id": "other", "doc_name": "other.pdf", "chunk_id": 1,
             "content": "Unrelated course content.", "page": 1, "slide": None},
        ],
        "course_topics": course_topics or [],
    })


LLM_TOPICS = {
    "topics": [
        {"name": "Hash Tables", "slug": "hash-tables", "position": 1,
         "description": "Hashing maps keys to buckets for O(1) lookup.",
         "doc_coverage": [{"doc": "hashing.pdf"}],           # pages backfilled
         "prereq_slugs": ["binary-search-trees"]},
        {"name": "Binary Search Trees", "slug": "Binary Search Trees",  # needs slugify
         "position": 0,
         "description": "Ordered trees supporting logarithmic search.",
         "doc_coverage": [{"doc": "trees.pdf", "pages": [2, 9]},
                          {"doc": "nonexistent.pdf", "pages": [1, 5]}],
         "prereq_slugs": ["not-a-real-topic"]},              # filtered out
        {"name": "Duplicate BSTs", "slug": "binary-search-trees", "position": 2,
         "description": "Duplicate slug must be dropped."},
    ]
}


@pytest.fixture
def brain_db(monkeypatch):
    def _install(course_topics=None, llm=LLM_TOPICS, llm_should_run=True):
        db = _content_db(course_topics)
        monkeypatch.setattr(course_brain, "_supabase", db)

        def fake_structured_call(*_args, **_kwargs):
            if not llm_should_run:
                raise AssertionError("structured_call must not be invoked")
            return llm

        monkeypatch.setattr(course_brain, "structured_call", fake_structured_call)
        return db

    return _install


# ---- synthesis --------------------------------------------------------------

def test_synthesize_persists_cleaned_topics(brain_db):
    db = brain_db()
    topics = course_brain.synthesize_topics(COURSE)

    # Sorted by position and reindexed; duplicate slug dropped.
    assert [t.slug for t in topics] == ["binary-search-trees", "hash-tables"]
    assert [t.position for t in topics] == [0, 1]

    bst, hashing = topics
    # Bogus doc filtered from coverage; missing pages backfilled from chunks.
    assert bst.doc_coverage == ({"doc": "trees.pdf", "pages": [2, 9]},)
    assert hashing.doc_coverage == ({"doc": "hashing.pdf", "pages": [1, 1]},)
    # Prereqs restricted to slugs within the topic set.
    assert bst.prereq_slugs == ()
    assert hashing.prereq_slugs == ("binary-search-trees",)

    rows = db.tables["course_topics"]
    assert {r["slug"] for r in rows} == {"binary-search-trees", "hash-tables"}
    assert all(r["course_id"] == COURSE for r in rows)


def test_synthesize_is_idempotent_rebuild(brain_db):
    db = brain_db(course_topics=[
        {"course_id": COURSE, "slug": "stale-topic", "name": "Stale Topic",
         "description": "", "doc_coverage": [], "prereq_slugs": [], "position": 0},
        {"course_id": "other", "slug": "keep-me", "name": "Keep Me",
         "description": "", "doc_coverage": [], "prereq_slugs": [], "position": 0},
    ])
    course_brain.synthesize_topics(COURSE)
    rows = db.tables["course_topics"]
    slugs_for_course = {r["slug"] for r in rows if r["course_id"] == COURSE}
    assert "stale-topic" not in slugs_for_course           # replaced wholesale
    assert slugs_for_course == {"binary-search-trees", "hash-tables"}
    assert any(r["slug"] == "keep-me" for r in rows)       # other course untouched


def test_synthesize_no_content_returns_empty_without_llm(monkeypatch):
    monkeypatch.setattr(course_brain, "_supabase",
                        FakeSupabase({"files": [], "embeddings": [], "course_topics": []}))
    monkeypatch.setattr(course_brain, "structured_call",
                        lambda *a, **k: pytest.fail("LLM must not be called"))
    assert course_brain.synthesize_topics("empty-course") == []


def test_synthesize_unusable_llm_output_raises(brain_db):
    brain_db(llm={"topics": [{"name": "", "slug": "", "description": "", "position": 0}]})
    with pytest.raises(RuntimeError):
        course_brain.synthesize_topics(COURSE)


# ---- reads -------------------------------------------------------------------

def test_get_topics_reads_table_without_llm(brain_db):
    brain_db(course_topics=[
        {"course_id": COURSE, "slug": "b-topic", "name": "B Topic",
         "description": "", "doc_coverage": [], "prereq_slugs": [], "position": 1},
        {"course_id": COURSE, "slug": "a-topic", "name": "A Topic",
         "description": "", "doc_coverage": [], "prereq_slugs": [], "position": 0},
    ], llm_should_run=False)
    topics = course_brain.get_topics(COURSE)
    assert [t.slug for t in topics] == ["a-topic", "b-topic"]   # position order


def test_get_topics_auto_generates_when_table_empty(brain_db):
    db = brain_db()
    topics = course_brain.get_topics(COURSE, auto_generate=True)
    assert [t.slug for t in topics] == ["binary-search-trees", "hash-tables"]
    assert db.tables["course_topics"]                            # persisted


def test_get_topics_no_auto_generate_returns_empty(brain_db):
    brain_db(llm_should_run=False)
    assert course_brain.get_topics(COURSE, auto_generate=False) == []


def test_topic_names_returns_ordered_names(brain_db):
    brain_db()
    assert course_brain.topic_names(COURSE) == ["Binary Search Trees", "Hash Tables"]


# ---- matching ---------------------------------------------------------------

TOPICS = [
    Topic(slug="binary-search-trees", name="Binary Search Trees", position=0),
    Topic(slug="hash-tables", name="Hash Tables", position=1),
]


def test_slugify():
    assert slugify("Binary Search Trees!") == "binary-search-trees"


def test_match_topic_exact_name_case_insensitive():
    assert match_topic("  binary search trees ", TOPICS).slug == "binary-search-trees"


def test_match_topic_by_slug():
    assert match_topic("Hash Tables?!", TOPICS).slug == "hash-tables"


def test_match_topic_substring_both_ways():
    assert match_topic("Trees", TOPICS).slug == "binary-search-trees"
    assert match_topic("Advanced Hash Tables Deep Dive", TOPICS).slug == "hash-tables"


def test_match_topic_no_match_returns_none():
    assert match_topic("Quantum Physics", TOPICS) is None
    assert match_topic("", TOPICS) is None


def test_match_mastery_exact_and_substring():
    assert match_mastery("Mitosis", {"mitosis": 0.8}) == 0.8
    assert match_mastery("Cell Division", {"division": 0.5}) == 0.5


def test_match_mastery_bridges_legacy_filename_labels_via_tokens():
    # Old filename-era rows keep matching new Course Brain names.
    legacy = {"301 3 excel": 0.7, "trees part1 something": 0.4}
    assert match_mastery("Excel Formulas", legacy) == 0.7
    assert match_mastery("Binary Trees", legacy) == 0.4


def test_match_mastery_prefers_most_specific_substring():
    mastery = {"trees": 0.2, "binary search trees": 0.9}
    assert match_mastery("Binary Search Trees Overview", mastery) == 0.9


def test_match_mastery_no_match_returns_none():
    assert match_mastery("Physics", {"chemistry": 0.1}) is None
    assert match_mastery("anything", {}) is None


def test_match_mastery_rows_bridges_rows():
    rows = [
        {"topic": "301 3 Excel", "mastery_level": 0.65},
        {"topic": "Irrelevant", "mastery_level": 0.1},
        {"topic": None, "mastery_level": 0.9},
        {"topic": "No Level"},
    ]
    assert match_mastery_rows("Excel Formulas", rows) == 0.65
    assert match_mastery_rows("Excel Formulas", []) is None
