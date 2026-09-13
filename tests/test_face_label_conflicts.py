"""Label-conflict detection: clusters holding 2+ named people."""
import pytest
from photosearch.face_review import find_label_conflicts


def test_ranks_most_lopsided_first():
    """11-vs-2 is a near-certain mislabel; 3-vs-2 is more likely a real merge."""
    clusters = {}
    labels = {}
    for i in range(11):
        clusters[i] = 0; labels[i] = 'Beckham'
    for i in range(11, 13):
        clusters[i] = 0; labels[i] = 'Carson'
    for i in range(20, 23):
        clusters[i] = 1; labels[i] = 'Asa'
    for i in range(23, 25):
        clusters[i] = 1; labels[i] = 'Levon'
    out = find_label_conflicts(clusters, labels)
    assert [c['cluster'] for c in out] == [0, 1]
    assert out[0]['lopsidedness'] > out[1]['lopsidedness']
    assert out[0]['labels'][0]['name'] == 'Beckham'   # majority first
    assert out[0]['labels'][1]['face_ids'] == [11, 12]


def test_pure_and_noise_clusters_are_not_conflicts():
    clusters = {1: 0, 2: 0, 3: -1, 4: -1}
    labels = {1: 'Asa', 2: 'Asa', 3: 'Asa', 4: 'Levon'}   # 3/4 are NOISE
    assert find_label_conflicts(clusters, labels) == []


def test_unnamed_faces_are_carried_not_counted():
    """Unnamed members ride along so they can be fixed in the same pass, but
    they must not affect the lopsidedness ranking."""
    clusters = {1: 0, 2: 0, 3: 0, 9: 0}
    labels = {1: 'Asa', 2: 'Asa', 3: 'Levon'}            # 9 unnamed
    out = find_label_conflicts(clusters, labels)
    assert out[0]['named_total'] == 3
    assert out[0]['unnamed_face_ids'] == [9]
    assert out[0]['size'] == 4


def test_no_majority_suggestion_is_emitted():
    """Deliberate: auto-applying the majority name would have been WRONG on the
    case that motivated this (both labels were bad; the answer was a third
    person). The payload must present labels, never a recommendation."""
    clusters = {1: 0, 2: 0, 3: 0}
    labels = {1: 'Beckham', 2: 'Beckham', 3: 'Carson'}
    c = find_label_conflicts(clusters, labels)[0]
    assert 'suggested_name' not in c and 'apply' not in c


def test_scope_is_required(client):
    """Unscoped clustering is minutes long and holds the NAS write lock."""
    r = client.get('/api/faces/label-conflicts')
    assert r.status_code == 400
    assert 'scope' in r.json()['detail'].lower()


def test_scoped_call_returns_shape(client):
    r = client.get('/api/faces/label-conflicts?date_from=2020-01-01&date_to=2030-01-01')
    assert r.status_code == 200
    d = r.json()
    assert 'conflicts' in d and 'scope_faces' in d and d['eps'] == 0.80
