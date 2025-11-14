one_shot_ner_paragraph2 = """{
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

one_shot_ner_output2 = """{
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

one_shot_ner_paragraph = """{
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

one_shot_ner_output = """{
    "output": []
}
"""

one_shot_ner_paragraph3 = """{
    "sentence": "杰里·贝勒斯（Jerryd Bayless），1988年8月20日出生于美国亚利桑那州菲尼克斯（Phoenix, Arizona），美国职业篮球运动员，司职后卫，效力于NBA费城76人队",
    "schema": [
      "丈夫",
      "朝代",
      "祖籍",
      "国籍"
    ],
    "coarse_types": [
      "人",
      "产品",
      "位置",
      "时间",
      "组织机构"
    ]
}
"""

one_shot_ner_output3 = """{
    "output": [
      {
        "subject": [
          "杰里·贝勒斯",
          "人",
          "运动员"
        ],
        "relationship": "国籍",
        "object": [
          "美国",
          "位置",
          "国家"
        ]
      }
    ]
}
"""