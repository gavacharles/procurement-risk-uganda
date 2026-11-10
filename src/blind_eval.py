"""Blind bid evaluation and private conflict-of-interest checks (number theory / finite fields).

Mathematics
-----------
Work in the multiplicative group of the prime field GF(p), where p is the 2048-bit
safe prime of RFC 3526 (group 14), p = 2q + 1 with q prime. We use the subgroup of
quadratic residues, which has prime order q.

  * Hash-to-group:   h(x) = SHA-256(x)^2 mod p        (squaring lands in the QR subgroup)
  * Blinding:        E_k(y) = y^k mod p,  k ∈ [2, q-1] secret
  * Commutativity:   E_a(E_b(y)) = y^(ab) = E_b(E_a(y))  (exponents commute)
  * Security rests on the Decisional Diffie–Hellman assumption in this subgroup: without k,
    E_k(h(x)) reveals nothing about x; with k, decryption needs k^-1 mod q.

Uses
----
1. Blind evaluation. PPDA holds key k. Each bidder becomes a code
   CODE = codename(E_k(h(bidder_id ‖ tender_id))). The committee sees codes only; names,
   logos and company identifiers are stripped. The code differs per tender, so the
   committee cannot link a firm across tenders. After evaluation PPDA reveals the mapping.

2. Private conflict-of-interest check (private set intersection).
   PPDA has bidders B; an ethics office has firms linked to committee members C
   (from declarations or the URSB register). Neither should see the other's list.
     PPDA sends {E_a(h(b))}; ethics office returns {E_b(E_a(h(b)))} and sends {E_b(h(c))};
     PPDA computes {E_a(E_b(h(c)))}. Equal double-blinded values ⇔ same firm.
   Only the overlap is learned (by PPDA) and nothing else.

This is a reference implementation for research/pilots; production use needs key
management, an audited crypto library, and a legal basis under the Data Protection
and Privacy Act 2019.
"""
from __future__ import annotations

import hashlib
import secrets

# RFC 3526, 2048-bit MODP group (group 14): safe prime p = 2q + 1
P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
Q = (P - 1) // 2

ADJ = ["amber", "azure", "cedar", "coral", "ember", "ivory", "jade", "lunar", "maple", "ochre",
       "onyx", "pearl", "quartz", "sable", "slate", "teal", "umber", "violet", "willow", "zephyr"]
NOUN = ["acacia", "baobab", "crane", "delta", "egret", "falcon", "gazelle", "heron", "ibis", "kob",
        "lake", "mango", "nile", "okapi", "papyrus", "ridge", "savanna", "tilapia", "valley", "zebra"]


def keygen() -> int:
    return 2 + secrets.randbelow(Q - 3)


def h(x: str) -> int:
    v = int.from_bytes(hashlib.sha256(x.encode()).digest(), "big") % P
    return pow(v, 2, P)  # quadratic residue


def blind(y: int, k: int) -> int:
    return pow(y, k, P)


def unblind(y: int, k: int) -> int:
    return pow(y, pow(k, -1, Q), P)


def codename(y: int) -> str:
    d = hashlib.sha256(y.to_bytes(256, "big")).digest()
    return f"{ADJ[d[0] % len(ADJ)].title()}-{NOUN[d[1] % len(NOUN)].title()}-{d[2:4].hex().upper()}"


def blind_codes(bidders: list[str], tender_id: str, k: int) -> dict[str, str]:
    """Bidder -> anonymous code for one tender (kept by PPDA; committee sees codes only)."""
    return {b: codename(blind(h(f"{b}|{tender_id}"), k)) for b in bidders}


def psi(bidders: list[str], related_firms: list[str]) -> set[str]:
    """Two-party private set intersection via commutative blinding (both keys simulated here)."""
    a, b = keygen(), keygen()
    ppda_sent = {blind(h(x), a): x for x in bidders}                  # PPDA -> ethics office
    returned = {y: blind(y, b) for y in ppda_sent}                    # ethics office -> PPDA
    ethics_sent = {blind(h(c), b) for c in related_firms}             # ethics office -> PPDA
    double_ethics = {blind(y, a) for y in ethics_sent}                # PPDA finishes
    return {ppda_sent[y] for y, yy in returned.items() if yy in double_ethics}


if __name__ == "__main__":
    k = keygen()
    bidders = ["Firm Alpha Ltd", "Firm Beta Ltd", "Firm Gamma Ltd"]
    print("Blind codes, tender T-001:", blind_codes(bidders, "T-001", k))
    print("Blind codes, tender T-002:", blind_codes(bidders, "T-002", k))
    y = h("Firm Alpha Ltd")
    a, b = keygen(), keygen()
    assert blind(blind(y, a), b) == blind(blind(y, b), a), "commutativity"
    assert unblind(blind(y, a), a) == y, "invertibility"
    print("Conflict-of-interest overlap:", psi(bidders, ["Firm Beta Ltd", "Unrelated Co"]))
