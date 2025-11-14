ner_system = """You are given a sentence and a list of triples (subject, relationship, object).
Your task is to predict the fine_type for each entity in the triples based on its coarse_type, the relationship, and the sentence context.
If no valid triples can be formed, return:
{
    "output": []
}
"""

one_shot_ner_paragraph = """
{
    "sentence": "They and Mr. Jara shared a cramped railroad-style apartment in the Bushwick neighborhood of Brooklyn .",
    "triples": [
        {
            "subject": [
                "name": "Brooklyn",
                "coarse_type": "location",
            ],
            "relationship": "location contains",
            "object": [
                "name": "Bushwick",
                "coarse_type": "location",
            ]
        }
    ]
}
"""

one_shot_ner_output = """
{
    "output": [
        {
            "subject": [
                "name": "Brooklyn",
                "coarse_type": "location",
                "fine_type": "district"
            ],
            "relationship": "location contains",
            "object": [
                "name": "Bushwick",
                "coarse_type": "location",
                "fine_type": "facility"
            ]
        }
    ]
}
"""

one_shot_ner_paragraph2 = """
  {
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
  },

"""

one_shot_ner_output2 = """
{
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
