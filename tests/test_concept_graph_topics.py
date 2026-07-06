"""concept_graph — graph derived from Course Brain topics (hermetic).

The V3 graph has no LLM call of its own: nodes are course_topics rows, edges
come from prereq_slugs, and mastery overlays via Course Brain label bridging.
"""
import pytest

import concept_graph
import course_brain
from fakes_endpoints import FakeSupabase

pytestmark = pytest.mark.unit

COURSE = "cs101"
USER = "user-1"

TOPIC_ROWS = [
    {"course_id": COURSE, "slug": "arrays", "name": "Arrays",
     "description": "", "doc_coverage": [], "prereq_slugs": [], "position": 0},
    {"course_id": COURSE, "slug": "binary-search-trees", "name": "Binary Search Trees",
     "description": "", "doc_coverage": [], "prereq_slugs": ["arrays"], "position": 1},
    {"course_id": COURSE, "slug": "hash-tables", "name": "Hash Tables",
     "description": "", "doc_coverage": [],
     "prereq_slugs": ["arrays", "ghost-slug"], "position": 2},   # ghost ignored
]


@pytest.fixture
def graph_env(monkeypatch):
    def _install(topics=TOPIC_ROWS, progress=None):
        monkeypatch.setattr(course_brain, "_supabase",
                            FakeSupabase({"course_topics": list(topics)}))
        monkeypatch.setattr(concept_graph, "_supabase",
                            FakeSupabase({"learning_progress": progress or []}))
        monkeypatch.setattr(course_brain, "structured_call",
                            lambda *a, **k: pytest.fail("LLM must not be called"))

    return _install


def test_get_graph_derives_nodes_and_edges_from_topics(graph_env):
    graph_env()
    graph = concept_graph.get_graph(COURSE)
    assert graph["concepts"] == ["Arrays", "Binary Search Trees", "Hash Tables"]
    assert graph["edges"] == [
        {"prerequisite": "Arrays", "concept": "Binary Search Trees"},
        {"prerequisite": "Arrays", "concept": "Hash Tables"},
    ]


def test_get_graph_returns_none_without_topics(graph_env):
    graph_env(topics=[])
    assert concept_graph.get_graph(COURSE) is None


def test_build_graph_resynthesizes_topics(graph_env, monkeypatch):
    graph_env(topics=[])
    monkeypatch.setattr(course_brain, "synthesize_topics", lambda cid: [
        course_brain.Topic(slug="a", name="A", position=0),
        course_brain.Topic(slug="b", name="B", prereq_slugs=("a",), position=1),
    ])
    graph = concept_graph.build_graph(COURSE)
    assert graph == {"concepts": ["A", "B"],
                     "edges": [{"prerequisite": "A", "concept": "B"}]}


def test_build_graph_raises_when_no_content(graph_env, monkeypatch):
    graph_env(topics=[])
    monkeypatch.setattr(course_brain, "synthesize_topics", lambda cid: [])
    with pytest.raises(RuntimeError):
        concept_graph.build_graph(COURSE)


def test_graph_with_mastery_overlays_and_finds_blockers(graph_env):
    graph_env(progress=[
        # Legacy label bridges to 'Arrays' via substring/token matching.
        {"user_id": USER, "course_id": COURSE, "topic": "arrays part1", "mastery_level": 0.2},
        {"user_id": USER, "course_id": COURSE, "topic": "hash tables", "mastery_level": 0.3},
    ])
    out = concept_graph.graph_with_mastery(COURSE, USER)
    assert out["exists"] is True

    by_name = {n["concept"]: n for n in out["concepts"]}
    assert by_name["Arrays"] == {"concept": "Arrays", "mastery_pct": 20.0, "has_data": True}
    assert by_name["Hash Tables"]["mastery_pct"] == 30.0
    assert by_name["Binary Search Trees"]["has_data"] is False

    # Hash Tables is weak and its prerequisite Arrays is also weak -> blocker.
    assert {"concept": "Hash Tables", "prerequisite": "Arrays",
            "concept_pct": 30.0, "prerequisite_pct": 20.0} in out["blockers"]


def test_graph_with_mastery_no_topics_and_failed_build_reports_not_exists(graph_env, monkeypatch):
    graph_env(topics=[])
    monkeypatch.setattr(course_brain, "synthesize_topics", lambda cid: [])
    out = concept_graph.graph_with_mastery(COURSE, USER)
    assert out == {"concepts": [], "edges": [], "blockers": [], "exists": False}
