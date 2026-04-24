"""
pytest tests for CarDiagnosticNetwork
======================================
Run from the backend/ directory:
  pip install pytest
  pytest tests/test_network.py -v

These tests verify:
  1. Network builds and validates without errors.
  2. CPDs are mathematically valid (probabilities sum to 1).
  3. Inference is directionally correct (specific symptoms implicate correct faults).
  4. Empty symptoms returns priors.
  5. All fault nodes are returned in diagnosis results.
  6. Differential diagnosis returns a valid symptom suggestion.
  7. engine_misfire is correctly classified as a symptom, not a fault.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bayesian_network import CarDiagnosticNetwork, FAULT_NODES, SYMPTOM_NODES


@pytest.fixture(scope='module')
def net():
    """Build the network once for all tests — it's slow to construct."""
    return CarDiagnosticNetwork()


# ── Test 1: Network validity ──────────────────────────────────────────────────

def test_network_builds_and_validates(net):
    """Network must pass pgmpy's internal consistency check."""
    assert net.model is not None
    assert net.model.check_model() is True


# ── Test 2: CPD probability sums ─────────────────────────────────────────────

def test_all_cpds_are_valid(net):
    """Every CPD must have columns that sum to 1 within floating-point tolerance."""
    import numpy as np
    for cpd in net.model.cpds:
        col_sums = cpd.get_values().sum(axis=0)
        assert np.allclose(col_sums, 1.0, atol=1e-6), \
            f"CPD for '{cpd.variable}' columns do not sum to 1: {col_sums}"


# ── Test 3: Battery symptoms → battery_dead ──────────────────────────────────

def test_battery_symptoms_increase_battery_dead_probability(net):
    """
    When all classic battery symptoms are observed, P(battery_dead) must be
    substantially higher than its prior of 0.08.
    """
    symptoms = {
        'battery_warning_light': 1,
        'dim_headlights':        1,
        'clicking_sound':        1,
        'engine_not_starting':   1,
    }
    probs = net.diagnose(symptoms)
    prior = net._get_prior_probabilities()['battery_dead']

    assert probs['battery_dead'] > prior, \
        f"Expected P(battery_dead) > prior {prior:.3f}, got {probs['battery_dead']:.3f}"
    assert probs['battery_dead'] > 0.5, \
        f"Expected P(battery_dead) > 0.5, got {probs['battery_dead']:.3f}"


# ── Test 4: Head gasket symptoms → head_gasket_failure ───────────────────────

def test_white_smoke_implicates_head_gasket(net):
    """White smoke is strongly associated with head gasket failure."""
    symptoms  = {'white_smoke': 1, 'warning_lights_all': 1}
    probs     = net.diagnose(symptoms)
    prior     = net._get_prior_probabilities()['head_gasket_failure']

    assert probs['head_gasket_failure'] > prior, \
        f"Expected posterior > prior {prior:.3f}, got {probs['head_gasket_failure']:.3f}"


# ── Test 5: Empty symptoms returns prior probabilities ───────────────────────

def test_empty_symptoms_returns_priors(net):
    """No evidence should give exactly the prior probabilities."""
    probs  = net.diagnose({})
    priors = net._get_prior_probabilities()
    assert probs == priors, "diagnose({}) should return exact prior probabilities"


# ── Test 6: All fault nodes present in results ────────────────────────────────

def test_all_faults_returned_in_diagnosis(net):
    """diagnose() must return a probability for every fault in FAULT_NODES."""
    probs = net.diagnose({'rough_idle': 1, 'check_engine_light': 1})
    for fault in FAULT_NODES:
        assert fault in probs, f"Fault '{fault}' missing from diagnosis results"
        assert 0.0 <= probs[fault] <= 1.0, \
            f"Probability for '{fault}' out of [0,1]: {probs[fault]}"


# ── Test 7: engine_misfire is a symptom, not a fault ─────────────────────────

def test_engine_misfire_is_symptom_not_fault():
    """
    engine_misfire was incorrectly listed as a fault in the original code.
    It is an observable intermediate node caused by spark_plugs_fouled.
    """
    assert 'engine_misfire' not in FAULT_NODES, \
        "engine_misfire should NOT be in FAULT_NODES — it is an observable symptom"
    assert 'engine_misfire' in SYMPTOM_NODES, \
        "engine_misfire should be in SYMPTOM_NODES"


# ── Test 8: Differential diagnosis returns valid symptom ─────────────────────

def test_next_best_symptom_returns_valid_id(net):
    """get_next_best_symptom() must return a symptom ID that exists in the network."""
    current_evidence = {'engine_not_starting': 1}
    symptom_id, gain = net.get_next_best_symptom(current_evidence, 'battery_dead')

    assert symptom_id is not None, "Expected a symptom suggestion, got None"
    assert symptom_id in SYMPTOM_NODES, \
        f"Suggested symptom '{symptom_id}' is not in SYMPTOM_NODES"
    assert symptom_id not in current_evidence, \
        "Should not suggest an already-observed symptom"
    assert gain >= 0.0, f"Information gain should be non-negative, got {gain}"


# ── Test 9: Recommendations sorted by probability ────────────────────────────

def test_recommendations_are_sorted_by_probability(net):
    """get_recommendations() must return faults in descending probability order."""
    probs = net.diagnose({'rough_idle': 1, 'check_engine_light': 1, 'excessive_vibration': 1})
    recs  = net.get_recommendations(probs)
    probs_in_order = [r['probability'] for r in recs]
    assert probs_in_order == sorted(probs_in_order, reverse=True), \
        "Recommendations not sorted by probability"


# ── Test 10: Network structure type detection ──────────────────────────────────

def test_network_structure_node_types_correct(net):
    """
    BUG FIX TEST: Original code used '"_" in node' and tagged every node as
    'symptom'. Verify that fault nodes are correctly typed as 'fault'.
    """
    structure = net.get_network_structure()
    node_types = {n['id']: n['type'] for n in structure['nodes']}

    for fault_id in FAULT_NODES:
        assert node_types.get(fault_id) == 'fault', \
            f"Node '{fault_id}' should be type 'fault', got '{node_types.get(fault_id)}'"

    for symptom_id in SYMPTOM_NODES:
        assert node_types.get(symptom_id) == 'symptom', \
            f"Node '{symptom_id}' should be type 'symptom', got '{node_types.get(symptom_id)}'"
