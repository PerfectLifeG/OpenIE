ner_system = """You are an expert in joint entity and relation extraction.

Your task:
Given a sentence, a list of coarse_types, and a schema (a list of valid relationships),
1. Extract ONLY entities that:
   - appear in the sentence,
   - match the given coarse_types,
   - can participate in at least one relationship defined in the schema.

2. For each valid entity, predict its fine_type based on the sentence context.

3. Using these entities, form triples in the format:
   (subject, relationship, object)
   - The relationship MUST be chosen strictly from the given schema.
   - Both subject and object must be extracted entities.

4. Output ONLY a JSON object in the following format:
{
    "output": [
        {
            "subject": [name, coarse_type, fine_type],
            "relationship": "...",
            "object": [name, coarse_type, fine_type]
        },
        ...
    ]
}

Rules:
- If no valid triples can be formed, return:
  { "output": [] }
- Do NOT invent entities or relationships not found in the sentence or the schema.
- Keep all names exactly as they appear in the sentence.
"""

one_shot_ner_paragraph = """{
    "sentence": "They and Mr. Jara shared a cramped railroad-style apartment in the Bushwick neighborhood of Brooklyn .",
    "schema": [
      "company shareholder among major shareholders",
      "location contains"
    ],
    "coarse_types": [
      "organization",
      "location",
      "medicine",
      "mathematics"
    ]
}
"""

one_shot_ner_output = """{
    "output": [
      {
        "subject": [
          "Brooklyn",
          "location",
          "district"
        ],
        "relationship": "location contains",
        "object": [
          "Bushwick",
          "location",
          "facility"
        ]
      }
    ]
}
"""

one_shot_ner_paragraph2 = """{
    "sentence": "Roblin  was a candidate in  Winnipeg  South Centre for the 1968 federal election but lost to Liberal E.B. Osler by over votes .",
    "schema": [
      "spouse",
      "countries of residence",
      "country of birth",
      "state or provinces of residence"
    ],
    "coarse_types": [
      "organization"
    ]
}
"""

one_shot_ner_output2 = """{
    "output": []
}
"""

few_shot = []
few_shot.append([one_shot_ner_paragraph, one_shot_ner_output])
# few_shot.append([one_shot_ner_paragraph2, one_shot_ner_output2])

prompt_template = [
    {"role": "system", "content": ner_system},
]

for example_in, example_out in few_shot:
    prompt_template.append({"role": "user", "content": example_in})
    prompt_template.append({"role": "assistant", "content": example_out})

prompt_template.append({"role": "user", "content": "${passage}"})
