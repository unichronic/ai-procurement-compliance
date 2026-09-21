"""
Certification/QCO logic (PS requirement: "suggest mandatory certification
requirements, e.g. BIS Product Certification, CRS, Hallmarking").

Deliberately NOT a 3-way if/electronics-else/jewellery-else hardcoded rule.
Certification obligations are tied to whether a Quality Control Order (QCO)
exists for that specific standard right now -- a live, growing regulatory
list (~187 orders / ~769 categories as of 2026), not a fixed enum. This
module reads that mapping from data so it can be refreshed independently
of code, and it must be able to say "no mandatory certification identified"
just as confidently as it flags one -- an always-yes system isn't running
real logic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

SCHEME_DESCRIPTIONS = {
    "ISI": "BIS Product Certification (ISI mark) — factory audit + lab testing required before licence grant (BIS Conformity Assessment Regulations, 2018).",
    "CRS": "Compulsory Registration Scheme — self-declaration after testing at a BIS-recognised lab, registered with an R-number. No factory audit required (Electronics & IT Goods (Requirements for Compulsory Registration) Order).",
    "Hallmark": "BIS Hallmarking — per-item traceability mark (HUID), not a per-manufacturer licence. Mandatory for gold; voluntary for silver as of 2026, but HUID marking becomes mandatory the moment a manufacturer opts in.",
}


class CertificationAdvisor:
    def __init__(self, standards: List[Dict[str, Any]], qco_orders: List[Dict[str, Any]]):
        self.by_id = {s["id"]: s for s in standards}
        self.orders_by_id = {o["id"]: o for o in qco_orders}

    def advise(self, standard_id: str) -> Dict[str, Any]:
        s = self.by_id.get(standard_id)
        if not s:
            return {"error": f"unknown standard id {standard_id}"}

        qco = s.get("qco")
        if not qco:
            # Absence of a QCO record means one of two very different things,
            # and conflating them is the most dangerous thing this module could
            # do. QCO data covers a fraction of a percent of the corpus, so for
            # almost every standard we simply have not looked -- and telling an
            # officer "certification is not legally required" on that basis
            # could remove a mandatory ISI requirement from a live tender.
            if not s.get("qco_checked"):
                return {
                    "standard_id": standard_id,
                    "mandatory_certification": None,
                    "certification_status": "unknown",
                    "scheme": None,
                    "message": (
                        f"Certification requirements for {s['number']} have NOT been "
                        f"checked against the Quality Control Order list. This is not "
                        f"a clearance: a QCO may exist. Verify against the BIS QCO "
                        f"notifications before issuing a tender."
                    ),
                }

            return {
                "standard_id": standard_id,
                "mandatory_certification": False,
                "certification_status": "no_qco_identified",
                "scheme": None,
                "message": (
                    f"No Quality Control Order was identified for products under "
                    f"{s['number']} when this record was compiled. The underlying "
                    f"data is hand-compiled and not verified against official BIS "
                    f"notifications, so confirm before relying on it."
                ),
            }

        scheme = qco.get("scheme")
        order = self.orders_by_id.get(qco.get("order_id")) if qco.get("order_id") else None

        if qco.get("mandatory") is False:
            return {
                "standard_id": standard_id,
                "mandatory_certification": False,
                "scheme": scheme,
                "message": qco.get("note", f"{scheme} certification is currently voluntary for this product."),
                "scheme_description": SCHEME_DESCRIPTIONS.get(scheme),
            }

        return {
            "standard_id": standard_id,
            "mandatory_certification": True,
            "scheme": scheme,
            "scheme_description": SCHEME_DESCRIPTIONS.get(scheme),
            "governing_order": order,
            "message": (
                f"Mandatory: {scheme} certification required under "
                f"{order['name'] if order else 'an applicable QCO'} before this product can be "
                f"manufactured, stored, sold, or imported (BIS Act 2016, Sec. 16-17; "
                f"non-compliance carries penal liability under Sec. 29)."
            ),
        }
