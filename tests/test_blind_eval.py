import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import blind_eval as be  # noqa: E402


def test_safe_prime_structure():
    assert be.P == 2 * be.Q + 1
    assert pow(2, be.P - 1, be.P) == 1  # Fermat check


def test_commutative_and_invertible():
    y = be.h("Firm X")
    a, b = be.keygen(), be.keygen()
    assert be.blind(be.blind(y, a), b) == be.blind(be.blind(y, b), a)
    assert be.unblind(be.blind(y, a), a) == y


def test_codes_unlinkable_across_tenders_and_stable_within():
    k = be.keygen()
    c1 = be.blind_codes(["A", "B"], "T1", k)
    c2 = be.blind_codes(["A", "B"], "T2", k)
    assert c1 == be.blind_codes(["A", "B"], "T1", k)
    assert c1["A"] != c2["A"]


def test_psi_finds_only_overlap():
    assert be.psi(["A", "B", "C"], ["C", "D"]) == {"C"}
    assert be.psi(["A"], ["B"]) == set()
