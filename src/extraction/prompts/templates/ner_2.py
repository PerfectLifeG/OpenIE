ner_system = """You are given a sentence, a set of coarse_types, and a list of previously extracted entities. Your task is to determine which entities correspond to the specified coarse_types and label each entity with its associated coarse_types.
Note that a single entity may correspond to multiple coarse_types if it is relevant to more than one category.
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
    ],
    "entities": ["Brooklyn", "Bushwick"]
}
"""


one_shot_ner_output = """    
    {
        "entities": [
          {
            "name": "triglycerides",
            "coarse_type": "science",
          },
          {
            "name": "metformin",
            "coarse_type": "science",
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