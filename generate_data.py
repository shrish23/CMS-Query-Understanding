import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DATA_FILE = Path("data/Medicare_Provider_and_Supplier_Taxonomy_Crosswalk_October_2025.csv")
OUTPUT_FILE = Path("synthetic_queries.csv")
MODEL_NAME = "Qwen/Qwen3-4B"
HF_TOKEN = os.getenv("HF_TOKEN")
QUERIES_PER_SPECIALTY = 30

# Custom column names for easy processing
COLUMN_NAMES = [
    "specialty_code_raw",
    "specialty_description",
    "taxonomy_code",
    "taxonomy_description",
]

# All Clinical specialities from the data
SPECIALTIES = [
    "Physician/General Practice",
    "Physician/General Surgery",
    "Physician/Allergy/ Immunology",
    "Physician/Otolaryngology",
    "Physician/Anesthesiology",
    "Physician/Cardiovascular Disease (Cardiology)",
    "Physician/Dermatology",
    "Physician/Family Practice",
    "Physician/Interventional Pain Management",
    "Physician/Gastroenterology",
    "Physician/Internal Medicine",
    "Physician/Osteopathic Manipulative Medicine",
    "Physician/Neurology",
    "Physician/Neurosurgery",
    "Speech Language Pathologist",
    "Physician/Obstetrics & Gynecology",
    "Physician/Hospice and Palliative Care",
    "Physician/Ophthalmology",
    "Oral Surgery (Dentist only)",
    "Physician/Orthopedic Surgery",
    "Clinical Cardiac Electrophysiology",
    "Physician/Pathology",
    "Physician/Sports Medicine",
    "Physician/Plastic and Reconstructive Surgery",
    "Physician/Physical Medicine and Rehabilitation",
    "Physician/Psychiatry",
    "Physician/Geriatric Psychiatry",
    "Physician/Colorectal Surgery (Proctology)",
    "Physician/Pulmonary Disease",
    "Physician/Diagnostic Radiology",
    "Physician/Thoracic Surgery",
    "Physician/Urology",
    "Chiropractic",
    "Physician/Nuclear Medicine",
    "Physician/Pediatric Medicine",
    "Physician/Geriatric Medicine",
    "Physician/Nephrology",
    "Physician/Hand Surgery",
    "Optometry",
    "Certified Nurse Midwife",
    "Certified Registered Nurse Anesthetist (CRNA)",
    "Physician/Infectious Disease",
    "Physician/Endocrinology",
    "Podiatry",
    "Nurse Practitioner",
    "Psychologist, Clinical",
    "Audiologist",
    "Physical Therapist in Private Practice",
    "Physician/Rheumatology",
    "Occupational Therapist in Private Practice",
    "Registered Dietitian or Nutrition Professional",
    "Physician/Pain Management",
    "Physician/Peripheral Vascular Disease",
    "Physician/Vascular Surgery",
    "Physician/Cardiac Surgery",
    "Physician/Addiction Medicine",
    "Licensed Clinical Social Worker",
    "Physician/Critical Care (Intensivists)",
    "Physician/Hematology",
    "Physician/Hematology-Oncology",
    "Physician/Preventive Medicine",
    "Physician/Maxillofacial Surgery",
    "Physician/Neuropsychiatry",
    "Certified Clinical Nurse Specialist",
    "Physician/Medical Oncology",
    "Physician/Surgical Oncology",
    "Physician/Radiation Oncology",
    "Physician/Emergency Medicine",
    "Physician/Interventional Radiology",
    "Optician",
    "Physician Assistant",
    "Physician/Gynecological Oncology",
    "Physician/Sleep Medicine",
    "Physician/Interventional Cardiology",
    "Dentist",
    "Physician/Hospitalist",
    "Physician/Advanced Heart Failure and Transplant Cardiology",
    "Physician/Medical Toxicology",
    "Hematopoietic Cell Transplantation and Cellular Therapy",
    "Medical Genetics and Genomics",
    "Undersea and Hyperbaric Medicine",
    "Micrographic Dermatologic Surgery",
    "Adult Congenital Heart Disease",
    "Marriage and Family Therapist",
    "Mental Health Counselor",
    "Licensed Professional Counselor",
]


def load_top_50() -> list[str]:
    '''Load the Medicare Provider and Supplier Taxonomy Crosswalk data, rank specialties by number of unique taxonomy codes/descriptions,
     and return the top 50 specialty descriptions.'''
    df = pd.read_csv(DATA_FILE, dtype=str)
    df.columns = COLUMN_NAMES
    for col in df.columns:
        df[col] = df[col].str.strip()

    df = df[df["specialty_description"].isin(SPECIALTIES)].copy()
    df["specialty_description"] = df["specialty_description"].str.split("/", n=1).str[-1]

    ranking = (
        df.groupby(["specialty_code_raw", "specialty_description"])
        .agg(
            taxonomy_code_count=("taxonomy_code", "nunique"),
            taxonomy_description_count=("taxonomy_description", "nunique"),
            row_count=("taxonomy_code", "count"),
        )
        .reset_index()
        .sort_values(
            by=[
                "taxonomy_code_count",
                "taxonomy_description_count",
                "row_count",
                "specialty_description",
            ],
            ascending=[False, False, False, True],
        )
    )

    return ranking.head(50)["specialty_description"].tolist()


def parse_queries(text: str) -> list[str]:
    '''Parse the raw text output from the model into a clean list of queries. Handles various formatting styles.'''
    lines = text.strip().split("\n")
    queries = []
    for line in lines:
        line = line.strip()
        # strip leading list markers: "1.", "1)", "-", "*", "•"
        line = re.sub(r"^[\d]+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line).strip()
        if line and len(line) > 3:
            queries.append(line)
    return queries


def generate_queries(model, tokenizer, specialty: str, n: int = QUERIES_PER_SPECIALTY) -> list[str]:
    '''Generate synthetic search queries for a given medical specialty using the provided model and tokenizer.'''
    prompt = (
        f"Generate {n + 5} realistic patient or caregiver search queries that should be classified as "
        f'"{specialty}". Use everyday language and write exactly what a real person might type into '
        f"a search engine or health chatbot. Focus on symptoms, concerns, or situations rather than "
        f"doctor names or specialties. Avoid medical jargon whenever possible. Keep each query short "
        f"(3-12 words). Output one query per line with no numbering or explanations."
    )
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.85,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    queries = parse_queries(response)
    return queries[:n] if len(queries) >= n else queries


def load_generation_model(model_name: str = MODEL_NAME, hf_token: str | None = HF_TOKEN):
    """Load and return the tokenizer/model pair used for synthetic query generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def generate_dataset(
    queries_per_specialty: int = QUERIES_PER_SPECIALTY,
    output_file: Path = OUTPUT_FILE,
    specialties: list[str] | None = None,
    model_name: str = MODEL_NAME,
    hf_token: str | None = HF_TOKEN,
    progress_callback=None,
) -> pd.DataFrame:
    """Generate and persist the synthetic query dataset."""
    specialties = specialties or load_top_50()
    if progress_callback:
        progress_callback("loading_model", 0, len(specialties), f"Loading model {model_name}")
    tokenizer, model = load_generation_model(model_name=model_name, hf_token=hf_token)

    rows = []
    total = len(specialties)
    for i, specialty in enumerate(specialties, 1):
        if progress_callback:
            progress_callback("generating", i - 1, total, f"Generating queries for {specialty}")
        queries = generate_queries(model, tokenizer, specialty, n=queries_per_specialty)
        for query in queries:
            rows.append({"query": query, "target_class": specialty})

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    if progress_callback:
        progress_callback("complete", total, total, f"Saved {len(df)} rows to {output_file}")
    return df


def main() -> None:
    print(f"Loading taxonomy data from {DATA_FILE}")
    specialties = load_top_50()
    print(f"Top {len(specialties)} specialties loaded")

    print(f"\nLoading model {MODEL_NAME} ...")
    tokenizer, model = load_generation_model()
    print("Model ready\n")

    rows = []
    for i, specialty in enumerate(specialties, 1):
        print(f"[{i}/{len(specialties)}] {specialty}")
        queries = generate_queries(model, tokenizer, specialty)
        print(f"  → {len(queries)} queries generated")
        for q in queries:
            rows.append({"query": q, "target_class": specialty})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
