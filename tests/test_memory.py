import pytest

from jarvis.memory import Memory


def test_add_and_get(memory: Memory):
    mid = memory.add("The build uses make", kind="fact", tags="build")
    rec = memory.get(mid)
    assert rec is not None
    assert rec.content == "The build uses make"
    assert rec.kind == "fact"


def test_empty_content_rejected(memory: Memory):
    with pytest.raises(ValueError):
        memory.add("   ")


def test_search_finds_by_keyword(memory: Memory):
    memory.add("Deployment happens via GitHub Actions on tag push")
    memory.add("The API key lives in the environment")
    hits = memory.search("deployment")
    assert any("Deployment" in r.content for r in hits)


def test_search_multiword(memory: Memory):
    memory.add("Rate limiting is enforced per organization")
    hits = memory.search("rate limiting organization")
    assert len(hits) >= 1


def test_forget(memory: Memory):
    mid = memory.add("temporary note")
    assert memory.forget(mid) is True
    assert memory.get(mid) is None
    assert memory.forget(mid) is False


def test_list_and_count(memory: Memory):
    memory.add("one")
    memory.add("two")
    assert memory.count() == 2
    assert len(memory.list()) == 2


def test_digest(memory: Memory):
    assert memory.digest() == ""
    memory.add("something worth remembering")
    assert "something worth remembering" in memory.digest()
