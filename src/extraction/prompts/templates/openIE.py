ner_system = (
    "You are an expert in joint entity and relation extraction. "
    "Given a sentence, coarse_types, and a schema, extract only entities that can participate in the schema and match the coarse_types. "
    "Predict the fine_type of these entities, then form triples (subject, relationship, object) using them. "
    "Output in JSON, with each subject/object as [name, coarse_type, fine_type], and relationship from the schema."
)

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
few_shot.append([one_shot_ner_paragraph2, one_shot_ner_output2])

prompt_template = [
    {"role": "system", "content": ner_system},
]

for example_in, example_out in few_shot:
    prompt_template.append({"role": "user", "content": example_in})
    prompt_template.append({"role": "assistant", "content": example_out})

prompt_template.append({"role": "user", "content": "${passage}"})
