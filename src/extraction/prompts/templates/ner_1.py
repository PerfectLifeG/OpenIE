ner_system = """You are given a sentence and a set of coarse_types and schema. Your task is to identify and extract all entities mentioned in the sentence that are semantically related to these coarse_types and schema.
Note that a single entity may correspond to multiple coarse_types and schema if it is relevant to more than one category.
"""

one_shot_ner_paragraph = """
Extract all entities from the sentence that match the given coarse types and schema.
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
    ]
}
"""


one_shot_ner_output = """    
{
    "entities": ["Brooklyn", "Bushwick"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]