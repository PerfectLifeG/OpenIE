ner_system = """You are given a sentence, a list of entities with their coarse_types, and a schema.
Your task is to form valid triples (subject, relationship, object) by matching entities to the schema based on their coarse_types and the sentence context.
If no valid triples can be formed, return:
{
    "triples": []
}
"""

few_shot = []

one_shot_ner_paragraph = """You are given a sentence, a list of entities with their coarse_types, and a schema. 
Your task is to form valid triples (subject, relationship, object) by matching entities to the schema based on their coarse_types and the sentence context.
{
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
    ],
    "entities": ["Brooklyn", "Bushwick"]
}
"""


one_shot_ner_output = """
{
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


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]