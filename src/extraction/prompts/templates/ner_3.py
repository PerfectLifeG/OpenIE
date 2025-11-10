ner_system = """You are given a sentence, a list of entities, and their corresponding coarse_types. Your task is to predict the fine_type for each entity based on the given information.
"""

one_shot_ner_paragraph = """{
    "sentence": "The aim of this study was to explain this finding by examining factors potentially related to triglycerides in these women . RESEARCH DESIGN AND METHODSOf the 733 women randomized to metformin or insulin in the MiG trial , 432 ( 219 metformin and 213 insulin ) had fasting plasma triglycerides measured at enrollment and at 36 weeks .",
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


one_shot_ner_output = """    
    {
        "entities": [
          {
            "name": "triglycerides",
            "coarse_type": "science",
            "fine_type": "chemical"
          },
          {
            "name": "metformin",
            "coarse_type": "science",
            "fine_type": "chemical"
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