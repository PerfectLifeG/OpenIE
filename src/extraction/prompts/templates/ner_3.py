ner_system = """You are given a sentence and a list of triples (subject, relationship, object).
Your task is to predict the fine_type for each entity in the triples based on its coarse_type, the relationship, and the sentence context.
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

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]
