ner_system = """You are an intelligent named entity recognition assistant for both Chinese and English text.

Task:
- You will be given a paragraph in JSON format, containing an "sentence", and a list of possible coarse entity types in "coarse_types".
- Your goal is to extract all named entities mentioned in the sentence.
- For each entity, provide:
- "name": the exact text of the entity
- "coarse_type": choose one value from the provided "coarse_types"
- "fine_type": determine a more specific type based on context

Output format:
- Return a JSON object with:
- "entities": a list of objects, each with "name", "coarse_type", "fine_type"

Requirements:
- The output must be valid JSON.
- Every "coarse_type" must be one of the types provided in the input's "coarse_types".
- Include all entities mentioned in the sentence, in either Chinese or English.
"""

one_shot_ner_paragraph = """{
    "sentence": "The aim of this study was to explain this finding by examining factors potentially related to triglycerides in these women . RESEARCH DESIGN AND METHODSOf the 733 women randomized to metformin or insulin in the MiG trial , 432 ( 219 metformin and 213 insulin ) had fasting plasma triglycerides measured at enrollment and at 36 weeks .",
    "coarse_types": [
      "vehicle information",
      "medicine",
      "biology",
      "science",
      "restaurant information",
      "location",
      "politics",
      "music"
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