import json

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


one_shot_ner_paragraph4 = """{
    "sentence": "GOALS -- Above all , protecting autonomous status of northern Iraq , which Americans established with air protection in 1991 ; keeping militia intact ; regional control over oil resources ; inclusion of Kirkuk , an oil center , in their autonomous region .",
    "schema": [
      "country of capital",
      "geographic distribution"
    ],
    "coarse_types": [
      "medicine",
      "economics"
    ]
}
"""

one_shot_ner_output4 = """{
    "output": []
}
"""

one_shot_ner_paragraph3 = """{
    "sentence": "邯郸银行股份有限公司党委书记、董事长郑志瑛，秦皇岛市海港区北环路街道军工里社区党委书记、居民委员会主任孙爱静，保定市民族学校校长马惠斌，张家口市怀来县王家楼回族乡委员会宣传委员、统战委员梁洪梅，新乐市彭家庄回族乡党委书记牛永辉等5位模范代表出席见面会，现场讲述推动民族团结进步的感人事迹",
    "schema": [
      "注册资本",
      "祖籍",
      "董事长",
      "首都"
    ],
    "coarse_types": [
      "医学",
      "组织机构",
      "产品",
      "人",
      "时间"
    ]
}
"""

one_shot_ner_output3 = """{
    "output": [
      {
        "subject": [
          "邯郸银行",
          "组织机构",
          "公司"
        ],
        "relationship": "董事长",
        "object": [
          "郑志瑛",
          "人",
          "企业家"
        ]
      }
    ]
}
"""

one_shot_ner_paragraph5 = """{
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

one_shot_ner_output5 = """{
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

few_shot = [
    # {
    #     "sentence": json.loads(one_shot_ner_paragraph)["sentence"],
    #     "schema": json.loads(one_shot_ner_paragraph)["schema"],
    #     "coarse_types": json.loads(one_shot_ner_paragraph)["coarse_types"],
    #     "output": json.loads(one_shot_ner_output)["output"]
    # },
    # {
    #     "sentence": json.loads(one_shot_ner_paragraph2)["sentence"],
    #     "schema": json.loads(one_shot_ner_paragraph2)["schema"],
    #     "coarse_types": json.loads(one_shot_ner_paragraph2)["coarse_types"],
    #     "output": json.loads(one_shot_ner_output2)["output"]
    # },
    {
        "sentence": json.loads(one_shot_ner_paragraph3)["sentence"],
        "schema": json.loads(one_shot_ner_paragraph3)["schema"],
        "coarse_types": json.loads(one_shot_ner_paragraph3)["coarse_types"],
        "output": json.loads(one_shot_ner_output3)["output"]
    },
    {
        "sentence": json.loads(one_shot_ner_paragraph4)["sentence"],
        "schema": json.loads(one_shot_ner_paragraph4)["schema"],
        "coarse_types": json.loads(one_shot_ner_paragraph4)["coarse_types"],
        "output": json.loads(one_shot_ner_output4)["output"]
    },
    {
        "sentence": json.loads(one_shot_ner_paragraph5)["sentence"],
        "schema": json.loads(one_shot_ner_paragraph5)["schema"],
        "coarse_types": json.loads(one_shot_ner_paragraph5)["coarse_types"],
        "output": json.loads(one_shot_ner_output5)["output"]
    }
]