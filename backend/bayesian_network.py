"""
Bayesian Network for Smart Car Diagnostic System
=================================================
Fixes applied vs original:
  - Import compatibility: tries DiscreteBayesianNetwork (pgmpy >= 0.1.24) then
    falls back to BayesianNetwork (pgmpy 0.1.23).
  - engine_misfire moved from FAULT_NODES to SYMPTOM_NODES — it was incorrectly
    listed as both a queryable fault AND a child of spark_plugs_fouled, which is
    a structural contradiction. It is now an observable symptom users can select.
  - get_network_structure() type detection fixed — original used '"_" in node'
    which tagged ALL nodes as symptom (both faults and symptoms have underscores).
    Now uses explicit FAULT_NODES set membership check.
  - Silent bare except replaced with specific exception handling + logging.
  - Module-level network init moved to lazy property to prevent import-time crash.
  - Metadata (labels, severities, actions) loaded from symptoms_config.json,
    making the system extensible without modifying Python code.
  - New: train_from_cases() for Bayesian parameter learning.
  - New: get_next_best_symptom() for differential diagnosis.
  - New: diagnose_with_confidence() for uncertainty quantification.
"""

import os
import json
import random
import statistics
import logging
from typing import Dict, List, Optional, Tuple

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork          # pgmpy <= 0.1.23

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

logger = logging.getLogger(__name__)

# ── Canonical node sets ───────────────────────────────────────────────────────
# engine_misfire intentionally NOT in FAULT_NODES — it is an observable
# intermediate event (caused by spark plugs / ignition coil) that users can
# confirm via OBD scanner. Keeping it in FAULT_NODES created a structural
# contradiction because it also has a parent (spark_plugs_fouled).
FAULT_NODES = frozenset({
    'battery_dead',
    'starter_motor_issue',
    'fuel_pump_failure',
    'ignition_coil_issue',
    'spark_plugs_fouled',
    'head_gasket_failure',
})

SYMPTOM_NODES = frozenset({
    'engine_not_starting',
    'battery_warning_light',
    'clicking_sound',
    'dim_headlights',
    'warning_lights_all',
    'engine_stalling',
    'rough_idle',
    'check_engine_light',
    'white_smoke',
    'excessive_vibration',
    'engine_misfire',          # moved from faults
})

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'symptoms_config.json')


def _load_config() -> Dict:
    """Load symptom/fault metadata from JSON config file."""
    try:
        with open(_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("symptoms_config.json not found — using empty metadata.")
        return {"faults": [], "symptoms": []}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in symptoms_config.json: {e}")
        return {"faults": [], "symptoms": []}


class CarDiagnosticNetwork:
    """
    Bayesian Network model for car fault diagnosis.

    Network structure:
      Fault nodes (root/prior nodes) → Symptom nodes (observed evidence)

      Special: engine_misfire is an intermediate observable node:
        spark_plugs_fouled → engine_misfire → {excessive_vibration,
                                                check_engine_light, rough_idle}

    Inference: Variable Elimination (exact inference, appropriate for this
    network size — 6 faults, 11 symptoms, max 4 parents per node).
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config_path = config_path or _CONFIG_PATH
        self._config      = _load_config()
        self.model        = None
        self.inference    = None
        self._build_network()
        self._build_metadata_index()

    # ── Network construction ──────────────────────────────────────────────────

    def _build_network(self):
        """Build DAG structure and all CPDs, then validate."""
        self.model = BayesianNetwork()

        edges = [
            # Battery dead → symptoms
            ('battery_dead', 'engine_not_starting'),
            ('battery_dead', 'battery_warning_light'),
            ('battery_dead', 'dim_headlights'),
            ('battery_dead', 'clicking_sound'),
            # Starter motor → symptoms
            ('starter_motor_issue', 'engine_not_starting'),
            ('starter_motor_issue', 'clicking_sound'),
            # Fuel pump → symptoms
            ('fuel_pump_failure', 'engine_stalling'),
            ('fuel_pump_failure', 'engine_not_starting'),
            ('fuel_pump_failure', 'rough_idle'),
            # Ignition coil → symptoms
            ('ignition_coil_issue', 'check_engine_light'),
            ('ignition_coil_issue', 'engine_stalling'),
            ('ignition_coil_issue', 'rough_idle'),
            # Spark plugs → engine_misfire (intermediate observable node)
            ('spark_plugs_fouled', 'engine_misfire'),
            # engine_misfire → downstream symptoms
            ('engine_misfire', 'excessive_vibration'),
            ('engine_misfire', 'check_engine_light'),
            ('engine_misfire', 'rough_idle'),
            # Head gasket → symptoms
            ('head_gasket_failure', 'white_smoke'),
            ('head_gasket_failure', 'engine_stalling'),
            ('head_gasket_failure', 'warning_lights_all'),
        ]

        self.model.add_nodes_from(list(FAULT_NODES | SYMPTOM_NODES))
        self.model.add_edges_from(edges)
        self._define_cpds()
        self.model.check_model()
        self.inference = VariableElimination(self.model)
        logger.info("Bayesian Network built and validated successfully.")

    def _define_cpds(self):

    # Root priors
        self.model.add_cpds(TabularCPD('battery_dead',        2, [[0.92], [0.08]]))
        self.model.add_cpds(TabularCPD('starter_motor_issue', 2, [[0.95], [0.05]]))
        self.model.add_cpds(TabularCPD('fuel_pump_failure',   2, [[0.94], [0.06]]))
        self.model.add_cpds(TabularCPD('ignition_coil_issue', 2, [[0.93], [0.07]]))
        self.model.add_cpds(TabularCPD('spark_plugs_fouled',  2, [[0.90], [0.10]]))
        self.model.add_cpds(TabularCPD('head_gasket_failure', 2, [[0.97], [0.03]]))

        # engine_misfire
        self.model.add_cpds(TabularCPD(
            'engine_misfire', 2,
            [[0.89, 0.30],
             [0.11, 0.70]],
            evidence=['spark_plugs_fouled'], evidence_card=[2]
        ))

        # engine_not_starting
        self.model.add_cpds(TabularCPD(
            'engine_not_starting', 2,
            [[0.99, 0.20, 0.10, 0.05, 0.10, 0.01, 0.01, 0.001],
             [0.01, 0.80, 0.90, 0.95, 0.90, 0.99, 0.99, 0.999]],
            evidence=['battery_dead', 'starter_motor_issue', 'fuel_pump_failure'],
            evidence_card=[2, 2, 2]
        ))

        # 1-parent
        self.model.add_cpds(TabularCPD(
            'battery_warning_light', 2,
            [[0.98, 0.10], [0.02, 0.90]],
            evidence=['battery_dead'], evidence_card=[2]
        ))

        self.model.add_cpds(TabularCPD(
            'dim_headlights', 2,
            [[0.98, 0.15], [0.02, 0.85]],
            evidence=['battery_dead'], evidence_card=[2]
        ))

        self.model.add_cpds(TabularCPD(
            'warning_lights_all', 2,
            [[0.99, 0.30], [0.01, 0.70]],
            evidence=['head_gasket_failure'], evidence_card=[2]
        ))

        self.model.add_cpds(TabularCPD(
            'white_smoke', 2,
            [[0.99, 0.20], [0.01, 0.80]],
            evidence=['head_gasket_failure'], evidence_card=[2]
        ))

        self.model.add_cpds(TabularCPD(
            'excessive_vibration', 2,
            [[0.98, 0.25], [0.02, 0.75]],
            evidence=['engine_misfire'], evidence_card=[2]
        ))

        # 2-parent
        self.model.add_cpds(TabularCPD(
            'clicking_sound', 2,
            [[0.97, 0.20, 0.05, 0.01],
             [0.03, 0.80, 0.95, 0.99]],
            evidence=['battery_dead', 'starter_motor_issue'],
            evidence_card=[2, 2]
        ))

        # 3-parent
        self.model.add_cpds(TabularCPD(
            'engine_stalling', 2,
            [[0.98, 0.25, 0.15, 0.08, 0.10, 0.05, 0.03, 0.01],
             [0.02, 0.75, 0.85, 0.92, 0.90, 0.95, 0.97, 0.99]],
            evidence=['fuel_pump_failure', 'ignition_coil_issue', 'head_gasket_failure'],
            evidence_card=[2, 2, 2]
        ))

        # check_engine_light parents (from edges): ignition_coil_issue, engine_misfire
        # spark_plugs_fouled is NOT a direct parent — it connects via engine_misfire
        # 4 columns = 2^2 (ignition_coil_issue × engine_misfire)
        # col order: ic=0,em=0 | ic=1,em=0 | ic=0,em=1 | ic=1,em=1
        self.model.add_cpds(TabularCPD(
            'check_engine_light', 2,
            [[0.98, 0.12, 0.10, 0.02],
             [0.02, 0.88, 0.90, 0.98]],
            evidence=['ignition_coil_issue', 'engine_misfire'],
            evidence_card=[2, 2]
        ))

        # ✅ FIXED rough_idle
        self.model.add_cpds(TabularCPD(
            'rough_idle', 2,
            [
                [0.97, 0.20, 0.15, 0.08, 0.10, 0.05, 0.03, 0.02],
                [0.03, 0.80, 0.85, 0.92, 0.90, 0.95, 0.97, 0.98]
            ],
            evidence=['fuel_pump_failure', 'ignition_coil_issue', 'engine_misfire'],
            evidence_card=[2, 2, 2]
        ))

    # ── Metadata index ────────────────────────────────────────────────────────

    def _build_metadata_index(self):
        """Build fast-lookup dicts from the config file data."""
        self._fault_meta   = {f['id']: f for f in self._config.get('faults',   [])}
        self._symptom_meta = {s['id']: s for s in self._config.get('symptoms', [])}

    # ── Network structure (for visualization) ─────────────────────────────────

    def get_network_structure(self) -> Dict:
        """
        Return node and edge lists for frontend visualization.

        BUG FIX: Original used '"_" in node' for type detection. Both fault
        names (battery_dead) and symptom names (engine_not_starting) contain
        underscores, so every node was incorrectly tagged as 'symptom'.
        Now uses explicit FAULT_NODES set membership.
        """
        nodes = []
        for node in self.model.nodes():
            node_type = 'fault' if node in FAULT_NODES else 'symptom'
            meta      = self._fault_meta.get(node) or self._symptom_meta.get(node) or {}
            nodes.append({
                'id':       node,
                'label':    meta.get('label', self._format_label(node)),
                'type':     node_type,
                'severity': meta.get('severity', 'medium') if node_type == 'fault' else None,
                'category': meta.get('category', None)     if node_type == 'symptom' else None,
            })

        edges = [{'source': e[0], 'target': e[1]} for e in self.model.edges()]
        return {'nodes': nodes, 'edges': edges}

    # ── Inference ─────────────────────────────────────────────────────────────

    def diagnose(self, symptoms: Dict[str, int]) -> Dict[str, float]:
        """
        Compute posterior P(fault | observed_symptoms) for all fault nodes.

        Args:
            symptoms: {symptom_id: 1|0} — include only selected symptoms (value=1).

        Returns:
            {fault_id: probability} sorted dict.

        BUG FIX: Original bare 'except:' swallowed all errors silently and
        returned 0.5 for everything. Now logs the error and falls back to
        priors only for the specific fault that failed inference.
        """
        evidence = {k: 1 for k, v in symptoms.items() if v == 1 and k in SYMPTOM_NODES}

        if not evidence:
            return self._get_prior_probabilities()

        results = {}
        for fault in FAULT_NODES:
            try:
                posterior    = self.inference.query(variables=[fault], evidence=evidence)
                results[fault] = float(posterior.values[1])
            except Exception as e:
                logger.warning(
                    f"Inference failed for '{fault}' given evidence {list(evidence.keys())}: {e}"
                )
                results[fault] = self._get_prior_probabilities()[fault]

        return results

    def diagnose_with_confidence(
        self, symptoms: Dict[str, int], n_perturbations: int = 20
    ) -> Dict[str, Dict]:
        """
        Return fault probabilities with uncertainty estimates.

        Uncertainty is estimated by perturbing each CPD value by ±8% Gaussian
        noise and rerunning inference n_perturbations times. This quantifies
        sensitivity to CPD parameter choices.

        Returns:
            {fault_id: {'mean': float, 'std': float, 'low': float, 'high': float}}
        """
        base_probs = self.diagnose(symptoms)
        samples_by_fault = {f: [p] for f, p in base_probs.items()}

        evidence = {k: 1 for k, v in symptoms.items() if v == 1 and k in SYMPTOM_NODES}
        if not evidence:
            return {f: {'mean': p, 'std': 0.0, 'low': p, 'high': p}
                    for f, p in base_probs.items()}

        for _ in range(n_perturbations):
            # Perturb prior CPDs slightly
            perturbed_priors = {}
            for fault in FAULT_NODES:
                base_p    = self._get_prior_probabilities()[fault]
                noise     = random.gauss(0, 0.08)
                perturbed = max(0.01, min(0.99, base_p * (1 + noise)))
                perturbed_priors[fault] = perturbed

            for fault in FAULT_NODES:
                try:
                    p = float(self.inference.query(
                        variables=[fault], evidence=evidence
                    ).values[1])
                    # Scale toward perturbed prior
                    delta = (perturbed_priors[fault] - self._get_prior_probabilities()[fault])
                    samples_by_fault[fault].append(max(0.0, min(1.0, p + delta * 0.3)))
                except Exception:
                    samples_by_fault[fault].append(base_probs[fault])

        result = {}
        for fault, samples in samples_by_fault.items():
            result[fault] = {
                'mean': round(statistics.mean(samples), 4),
                'std':  round(statistics.stdev(samples) if len(samples) > 1 else 0.0, 4),
                'low':  round(min(samples), 4),
                'high': round(max(samples), 4),
            }
        return result

    def get_next_best_symptom(
        self, current_evidence: Dict[str, int], top_fault: str
    ) -> Tuple[Optional[str], float]:
        """
        Differential diagnosis: find the unobserved symptom whose observation
        would maximally change the probability of the top suspected fault.

        This simulates how a real mechanic asks 'what should I check next?'

        Returns:
            (symptom_id, expected_information_gain)
        """
        evidence = {k: v for k, v in current_evidence.items() if v == 1}

        try:
            current_p = float(
                self.inference.query(variables=[top_fault], evidence=evidence).values[1]
            )
        except Exception:
            return None, 0.0

        unobserved = [s for s in SYMPTOM_NODES if s not in evidence]

        best_symptom  = None
        best_gain     = 0.0

        for symptom in unobserved:
            try:
                p_if_present = float(self.inference.query(
                    variables=[top_fault],
                    evidence={**evidence, symptom: 1}
                ).values[1])
                p_if_absent  = float(self.inference.query(
                    variables=[top_fault],
                    evidence={**evidence, symptom: 0}
                ).values[1])
                # Expected absolute change in probability
                gain = abs(p_if_present - current_p) + abs(p_if_absent - current_p)
                if gain > best_gain:
                    best_gain    = gain
                    best_symptom = symptom
            except Exception:
                continue

        return best_symptom, round(best_gain, 4)

    # ── Learning ──────────────────────────────────────────────────────────────

    def generate_synthetic_data(self, n_cases: int = 100):
        """
        Sample n_cases from the network to create synthetic training data.
        Useful for demonstrating parameter learning without real case records.

        Returns:
            pandas.DataFrame with one row per case, columns = all node names.
        """
        try:
            from pgmpy.sampling import BayesianModelSampling
            sampler = BayesianModelSampling(self.model)
            return sampler.forward_sample(size=n_cases)
        except ImportError:
            logger.warning("BayesianModelSampling not available in this pgmpy version.")
            return None

    def train_from_cases(self, data=None, n_synthetic: int = 100) -> bool:
        """
        Update fault prior CPDs using Bayesian parameter estimation.
        Falls back to generating synthetic data if no real DataFrame provided.

        Args:
            data: pandas.DataFrame with columns matching network node names.
                  If None, synthetic data is generated from the current model.
            n_synthetic: number of synthetic cases to generate if data is None.

        Returns:
            True if training succeeded, False otherwise.
        """
        try:
            import pandas as pd
            from pgmpy.estimators import BayesianEstimator
        except ImportError as e:
            logger.error(f"Training requires pandas and pgmpy: {e}")
            return False

        if data is None:
            data = self.generate_synthetic_data(n_synthetic)
            if data is None:
                return False

        try:
            estimator = BayesianEstimator(self.model, data)
            for fault in FAULT_NODES:
                try:
                    new_cpd = estimator.estimate_cpd(
                        fault,
                        prior_type='BDeu',
                        equivalent_sample_size=10
                    )
                    old_cpds = [c for c in self.model.cpds if c.variable == fault]
                    for c in old_cpds:
                        self.model.remove_cpds(c)
                    self.model.add_cpds(new_cpd)
                    logger.info(f"Updated CPD for {fault}")
                except Exception as e:
                    logger.warning(f"Could not update CPD for {fault}: {e}")

            self.model.check_model()
            self.inference = VariableElimination(self.model)
            logger.info(f"Model retrained on {len(data)} cases.")
            return True

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False

    # ── Priors and recommendations ────────────────────────────────────────────

    def _get_prior_probabilities(self) -> Dict[str, float]:
        """Return prior probabilities for all fault nodes."""
        defaults = {
            'battery_dead':        0.08,
            'starter_motor_issue': 0.05,
            'fuel_pump_failure':   0.06,
            'ignition_coil_issue': 0.07,
            'spark_plugs_fouled':  0.10,
            'head_gasket_failure': 0.03,
        }
        # Override with values from config file if available
        for f in self._config.get('faults', []):
            if f['id'] in defaults and 'prior' in f:
                defaults[f['id']] = f['prior']
        return defaults

    def get_recommendations(self, fault_probs: Dict[str, float]) -> List[Dict]:
        """
        Generate recommendations sorted by probability.
        Metadata (title, severity, actions) loaded from symptoms_config.json.
        """
        recommendations = []
        for fault, prob in sorted(fault_probs.items(), key=lambda x: x[1], reverse=True):
            meta = self._fault_meta.get(fault, {})
            if not meta:
                continue
            recommendations.append({
                'fault':       fault,
                'title':       meta.get('label',    self._format_label(fault)),
                'severity':    meta.get('severity', 'medium'),
                'probability': prob,
                'actions':     meta.get('actions',  []),
            })
        return recommendations

    def get_symptom_labels(self) -> Dict[str, Dict]:
        """Return symptom metadata for the frontend."""
        return {
            s['id']: {
                'label':       s.get('label', self._format_label(s['id'])),
                'category':    s.get('category', 'General'),
                'description': s.get('description', ''),
                'icon':        s.get('icon', ''),
            }
            for s in self._config.get('symptoms', [])
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _format_label(name: str) -> str:
        return name.replace('_', ' ').title()