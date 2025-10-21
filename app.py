import os, json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from openai import OpenAI

# ENV requis :
# SUPABASE_URL, SUPABASE_ANON_KEY (ou SERVICE_ROLE)
# OPENAI_API_KEY

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Set SUPABASE_URL, SUPABASE_ANON_KEY (ou SERVICE_ROLE) et OPENAI_API_KEY dans l'env.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="CRFPA grader minimal")

class GradeInput(BaseModel):
    exercise_slug: str
    candidate_answer: str

def fetch_methodology():
    # Doc courant
    doc_resp = supabase.table("methodology_docs").select("version, content, format").eq("is_current", True).single().execute()
    if doc_resp.error:
        raise HTTPException(500, f"methodology_docs error: {doc_resp.error.message}")
    doc = doc_resp.data

    # Critères (25 pts)
    crit_resp = supabase.table("methodology_criteria")\
        .select("order_index, label, max_points")\
        .eq("version", doc["version"])\
        .order("order_index", desc=False).execute()
    if crit_resp.error:
        raise HTTPException(500, f"methodology_criteria error: {crit_resp.error.message}")
    criteria = crit_resp.data
    return doc, criteria

def fetch_rubric(slug: str):
    rub_resp = supabase.table("exercise_rubrics")\
        .select("exercise_slug, part_title_expected, subpart_title_expected, scope, subsection, item_label, points_max, order_bucket, version")\
        .eq("exercise_slug", slug)\
        .order("order_bucket", desc=False)\
        .order("subsection", desc=False)\
        .execute()
    if rub_resp.error:
        raise HTTPException(500, f"exercise_rubrics error: {rub_resp.error.message}")
    return rub_resp.data

@app.post("/grade")
def grade(payload: GradeInput):
    doc, criteria = fetch_methodology()
    rubric = fetch_rubric(payload.exercise_slug)

    # Prompt compact JSON
    system = "\n".join([
        "Tu es un correcteur CRFPA.",
        "Produis UNIQUEMENT du JSON valide suivant ce schéma:",
        "{"
        "\"note_methodologie_sur20\": number,"
        "\"details_methodologie\": ["
        "{\"order_index\": number, \"label\": string, \"points_awarded\": number, \"max_points\": number}"
        "],"
        "\"note_specifique_sur20\": number,"
        "\"details_specifique\": ["
        "{\"part_title_expected\": string, \"subpart_title_expected\": string, \"subsection\": string, "
        "\"item_label\": string, \"points_awarded\": number, \"points_max\": number}"
        "],"
        "\"note_finale_sur20\": number"
        "}",
        "Formule: note_finale = ((note_specifique_sur20 * 3) + note_methodologie_sur20) / 4, arrondie au dixième.",
        "Évalue strictement à partir des données fournies (critères méthodo et grille spécifique). N’invente aucun item."
    ])

    user_payload = {
        "candidate_answer": payload.candidate_answer,
        "methodology_text": doc["content"],
        "methodology_criteria": criteria,         # [{order_index,label,max_points}, ...]
        "specific_rubric_rows": rubric            # 8 colonnes
    }

    try:
        resp = openai.chat.completions.create(
            model="gpt-4o-mini",          # ou le modèle que tu utilises
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ]
        )
        txt = resp.choices[0].message.content.strip()
        result = json.loads(txt)
    except Exception as e:
        raise HTTPException(500, f"OpenAI error: {e}")

    return result

# Pour Render : respecter $PORT
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
