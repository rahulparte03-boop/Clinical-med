import json
import os
import re
from datetime import datetime

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Clinical Rx Copilot", page_icon="🩺", layout="wide")

SYSTEM_PROMPT = r"""
You are a clinician-facing clinical decision-support assistant. Your output is a DRAFT for review by a licensed doctor, never an autonomous prescription.

Core behavior:
1. First identify immediate red flags / instability and say what requires urgent stabilization, referral, or escalation before routine outpatient treatment.
2. Provide a prioritized differential diagnosis with short reasoning tied only to supplied data.
3. State the most likely working diagnosis only when supported; otherwise state that diagnosis is uncertain.
4. Recommend only investigations that could reasonably change diagnosis, disposition, or treatment; distinguish urgent from routine.
5. Give practical advice: disposition, monitoring, hydration/diet/activity where relevant, follow-up interval, and explicit return precautions.
6. If a medication plan is appropriate, present a DRAFT prescription using generic names and include: drug, dose, route, frequency, duration, indication, key contraindications/cautions, and monitoring.
7. Never invent symptoms, exam findings, investigation results, allergies, pregnancy status, kidney/liver function, or medication history.
8. If key dosing information is missing (especially age, weight when needed, allergy status, pregnancy status when relevant, renal/hepatic function when relevant, or interacting medicines), do NOT guess a dose. Put the missing information in missing_for_safe_prescribing and leave the affected medication dose as null.
9. Pediatrics: use weight-based dosing only when weight is supplied; include maximum dose when clinically relevant. Do not apply adult vital thresholds as normal pediatric ranges.
10. Pregnancy/lactation: flag when treatment or investigation choice may differ.
11. Renal/hepatic impairment: flag medicines requiring adjustment or avoidance.
12. Antibiotic stewardship: do not recommend antibiotics for likely viral/self-limited illness without a bacterial indication. State the suspected bacterial indication when recommending one.
13. Avoid duplicate drug classes and clinically important interactions. Explicitly flag NSAID risk, anticoagulant/antiplatelet issues, QT-prolonging combinations, sedative stacking, hypoglycemia risk, nephrotoxicity, and allergy cross-reactivity when relevant.
14. Do not recommend discharge when the supplied data suggests instability or a time-critical diagnosis.
15. Be concise and clinically oriented. Use uncertainty labels: high / moderate / low confidence.

Return STRICT JSON only, with this exact top-level structure:
{
  "triage": {
    "acuity": "emergency|urgent|routine|uncertain",
    "red_flags": ["..."],
    "immediate_actions": ["..."]
  },
  "differential": [
    {"diagnosis": "...", "likelihood": "high|moderate|low", "reason": "...", "against": "..."}
  ],
  "working_diagnosis": {"diagnosis": "...", "confidence": "high|moderate|low|uncertain", "reason": "..."},
  "investigations": {
    "urgent": [{"test": "...", "why": "..."}],
    "routine": [{"test": "...", "why": "..."}],
    "not_needed_now": [{"test": "...", "why": "..."}]
  },
  "advice": {
    "disposition": "...",
    "supportive_care": ["..."],
    "monitoring": ["..."],
    "follow_up": "...",
    "return_precautions": ["..."]
  },
  "draft_prescription": [
    {
      "drug": "...",
      "dose": "... or null",
      "route": "... or null",
      "frequency": "... or null",
      "duration": "... or null",
      "indication": "...",
      "cautions": ["..."],
      "monitoring": ["..."]
    }
  ],
  "missing_for_safe_prescribing": ["..."],
  "doctor_checklist_before_signing": ["..."]
}
"""


def num_or_none(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def local_safety_screen(case):
    """Conservative pre-model screen. Adult cutoffs are not used as pediatric normal ranges."""
    alerts = []
    age = num_or_none(case.get("age_years"))
    adult = age is not None and age >= 18

    spo2 = num_or_none(case.get("spo2_percent"))
    sbp = num_or_none(case.get("sbp_mmHg"))
    dbp = num_or_none(case.get("dbp_mmHg"))
    pr = num_or_none(case.get("pulse_per_min"))
    rbs = num_or_none(case.get("rbs_mg_dl"))

    if spo2 is not None and spo2 <= 90:
        alerts.append("SpO₂ ≤90%: assess immediately for hypoxemia and need for oxygen/escalation.")
    if rbs is not None and rbs < 54:
        alerts.append("RBS <54 mg/dL: severe hypoglycemia range—treat promptly and reassess.")
    if rbs is not None and rbs >= 400:
        alerts.append("RBS ≥400 mg/dL: evaluate urgently for hyperglycemic emergency/dehydration/ketosis.")

    if adult:
        if sbp is not None and sbp < 90:
            alerts.append("Adult SBP <90 mmHg: assess for shock/hypoperfusion.")
        if (sbp is not None and sbp >= 180) or (dbp is not None and dbp >= 120):
            alerts.append("Severely elevated adult BP: actively assess for acute target-organ injury before treating as routine hypertension.")
        if pr is not None and (pr < 40 or pr > 150):
            alerts.append("Marked adult bradycardia/tachycardia: assess rhythm, perfusion, and reversible causes urgently.")

    return alerts


def extract_json(text):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def render_list(title, items):
    st.markdown(f"#### {title}")
    if not items:
        st.caption("None listed")
    else:
        for item in items:
            st.markdown(f"- {item}")


st.title("🩺 Clinical Rx Copilot")
st.caption("Doctor-facing clinical decision support • Draft only • Final diagnosis and prescription require clinician review")

with st.sidebar:
    st.header("Safety & model")
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5.6"))
    st.info("Do not enter patient name, phone, address, Aadhaar, or other unnecessary identifiers.")
    st.warning("This MVP does not replace examination, ECG/imaging review, local protocols, or emergency escalation.")

st.subheader("1. Patient context")
c1, c2, c3, c4 = st.columns(4)
with c1:
    age = st.number_input("Age (years)", min_value=0.0, max_value=120.0, value=30.0, step=1.0)
with c2:
    sex = st.selectbox("Sex", ["Male", "Female", "Intersex/other", "Unknown"])
with c3:
    weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=0.0, step=0.5, help="Use 0 if unknown")
with c4:
    pregnancy = st.selectbox("Pregnancy/lactation", ["Not applicable", "No", "Pregnant", "Lactating", "Unknown"])

st.subheader("2. Vitals")
v1, v2, v3, v4, v5, v6 = st.columns(6)
with v1:
    sbp = st.number_input("SBP", min_value=0, max_value=300, value=120)
with v2:
    dbp = st.number_input("DBP", min_value=0, max_value=200, value=80)
with v3:
    pr = st.number_input("Pulse/min", min_value=0, max_value=250, value=80)
with v4:
    spo2 = st.number_input("SpO₂ %", min_value=0, max_value=100, value=98)
with v5:
    rbs = st.number_input("RBS mg/dL", min_value=0, max_value=1500, value=100)
with v6:
    temp = st.number_input("Temp °C", min_value=30.0, max_value=45.0, value=37.0, step=0.1)

st.subheader("3. Clinical details")
complaint = st.text_area("Chief complaint + duration", height=90, placeholder="Example: Fever for 5 days, headache, myalgia; no bleeding, no dyspnea")
history_exam = st.text_area("Relevant history + examination", height=120, placeholder="Onset/course, associated symptoms, hydration, chest/abdomen/CNS exam, danger signs, etc.")
investigations = st.text_area("Available investigations", height=120, placeholder="CBC, LFT/RFT, ECG, X-ray, USG, malaria/dengue tests, urinalysis, etc.")

c5, c6 = st.columns(2)
with c5:
    allergies = st.text_area("Drug/food allergies", placeholder="Write 'None known' only if actually checked")
    comorbidities = st.text_area("Comorbidities", placeholder="HTN, DM, CKD, CLD, asthma, seizure disorder, etc.")
with c6:
    current_meds = st.text_area("Current medicines", placeholder="Include anticoagulants, antiplatelets, insulin/OHA, steroids, etc.")
    renal_hepatic = st.text_area("Renal/hepatic status", placeholder="Creatinine/eGFR, liver disease, or 'unknown'")

case = {
    "age_years": age,
    "sex": sex,
    "weight_kg": None if weight == 0 else weight,
    "pregnancy_lactation": pregnancy,
    "sbp_mmHg": sbp,
    "dbp_mmHg": dbp,
    "pulse_per_min": pr,
    "spo2_percent": spo2,
    "rbs_mg_dl": rbs,
    "temperature_c": temp,
    "chief_complaint": complaint.strip(),
    "history_and_exam": history_exam.strip(),
    "available_investigations": investigations.strip(),
    "allergies": allergies.strip() or "not provided",
    "comorbidities": comorbidities.strip() or "not provided",
    "current_medications": current_meds.strip() or "not provided",
    "renal_hepatic_status": renal_hepatic.strip() or "not provided",
}

local_alerts = local_safety_screen(case)
if local_alerts:
    st.error("Pre-check found potentially urgent data:")
    for a in local_alerts:
        st.markdown(f"- {a}")

if st.button("Generate clinical draft", type="primary", use_container_width=True):
    if not complaint.strip():
        st.error("Chief complaint is required.")
        st.stop()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set. Add it to your environment or Streamlit secrets before running.")
        st.stop()

    with st.spinner("Generating clinician draft..."):
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": "Assess this case. Return strict JSON only.\n\nCASE:\n" + json.dumps(case, ensure_ascii=False, indent=2),
                    },
                ],
            )
            result = extract_json(response.output_text)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    triage = result.get("triage", {})
    acuity = str(triage.get("acuity", "uncertain")).upper()
    if acuity == "EMERGENCY":
        st.error(f"TRIAGE: {acuity}")
    elif acuity == "URGENT":
        st.warning(f"TRIAGE: {acuity}")
    else:
        st.info(f"TRIAGE: {acuity}")

    left, right = st.columns(2)
    with left:
        render_list("Red flags", triage.get("red_flags", []))
    with right:
        render_list("Immediate actions", triage.get("immediate_actions", []))

    st.markdown("### Differential diagnosis")
    for i, dx in enumerate(result.get("differential", []), start=1):
        st.markdown(
            f"**{i}. {dx.get('diagnosis', 'Unknown')}** — {str(dx.get('likelihood', 'uncertain')).upper()}  \n"
            f"Why: {dx.get('reason', '')}  \n"
            f"Against/uncertain: {dx.get('against', '')}"
        )

    wd = result.get("working_diagnosis", {})
    st.markdown("### Working diagnosis")
    st.write(f"**{wd.get('diagnosis', 'Uncertain')}** — confidence: **{wd.get('confidence', 'uncertain')}**")
    st.write(wd.get("reason", ""))

    inv = result.get("investigations", {})
    st.markdown("### Investigations")
    for label, key in [("Urgent", "urgent"), ("Routine / next", "routine"), ("Not needed now", "not_needed_now")]:
        st.markdown(f"#### {label}")
        items = inv.get(key, []) or []
        if not items:
            st.caption("None listed")
        for item in items:
            st.markdown(f"- **{item.get('test', '')}** — {item.get('why', '')}")

    advice = result.get("advice", {})
    st.markdown("### Advice & disposition")
    st.write(f"**Disposition:** {advice.get('disposition', '')}")
    render_list("Supportive care", advice.get("supportive_care", []))
    render_list("Monitoring", advice.get("monitoring", []))
    st.write(f"**Follow-up:** {advice.get('follow_up', '')}")
    render_list("Return precautions", advice.get("return_precautions", []))

    st.markdown("### Draft prescription — doctor must verify before signing")
    meds = result.get("draft_prescription", []) or []
    if not meds:
        st.caption("No medicines suggested.")
    for m in meds:
        dose = m.get("dose") or "DOSE WITHHELD — missing/uncertain data"
        route = m.get("route") or "—"
        frequency = m.get("frequency") or "—"
        duration = m.get("duration") or "—"
        st.markdown(
            f"**{m.get('drug', '')}**  \n"
            f"Dose: **{dose}** | Route: {route} | Frequency: {frequency} | Duration: {duration}  \n"
            f"Indication: {m.get('indication', '')}"
        )
        if m.get("cautions"):
            st.caption("Cautions: " + "; ".join(m.get("cautions", [])))
        if m.get("monitoring"):
            st.caption("Monitoring: " + "; ".join(m.get("monitoring", [])))
        st.divider()

    missing = result.get("missing_for_safe_prescribing", []) or []
    if missing:
        st.warning("Missing information for safe prescribing: " + "; ".join(missing))

    render_list("Doctor checklist before signing", result.get("doctor_checklist_before_signing", []))

    with st.expander("Raw JSON / audit copy"):
        st.json(result)
        st.caption(f"Generated {datetime.now().isoformat(timespec='seconds')} • model: {model}")

    st.warning("Clinical draft only. Reconcile with examination, local hospital protocol/formulary, allergies, pregnancy status, renal/hepatic function, current medicines, and your own clinical judgment before prescribing.")
