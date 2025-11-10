任务描述：我会给你一条句子和一个分类集合（coarse_types），你需要根据这个分类集合抽取句子中的名词，抽取时，名词的类型应该是分类集合中的任意类型。然后对抽取到的名词进行更加细节的分类（fine_type），要求 fine_type 决不能为空或 "unknown"，必须要判断唯一一个最相近的细分类，并且要使用与输入相同的语言。最终以 json 格式返回。 
如果句子中没有任何名词符合分类集合中的类型，返回一个空的 entities 列表。

示例1： 输入如下： 句子： “截至9月末，深圳现金累计投放量同比出现负数。”近日，一位接近监管部门人士对本报记者称，“

coarse_types： "生物", "职位", "科学", "组织机构", "学历", "位置"

那么输出应该如下： { “entities”（代表抽取到的所有名词）: [ { "name": "记者",（代表抽取到的其中一个名词） "coarse_type": "职位",（代表抽取到的名词的类型） "fine_type": "概念",（代表抽取到的名词的更加细节的类型，要求 fine_type 决不能为空或 "unknown"，必要要判断一个最相近的细分类，必须要使用与输入相同的语言，例如句子为中文，则判断得到的 fine_type 也应该为中文） } ]（这个示例中只抽取到了一个名词） }

示例2： 输入如下： 句子： Since C4 - C - N - PEG9 has relatively bigger headgroup size compared to the C12EO4 , addition of C4 - C - N - PEG9 into wormlike micelles reduces the critical packing parameter resulting in the formation of spherical aggregates .

coarse_types： "biology","science","film information","restaurant information","product"

那么输出应该如下： { "entities": [ { "name": "C4 - C - N - PEG9", "coarse_type": "science", "fine_type": "chemical" }, { "name": "C12EO4", "coarse_type": "science", "fine_type": "chemical" } ]（这个示例中抽取到了2个名词） }

示例3：输入如下：句子：这不仅可以促进学术交流，同时也可以提高酒家的文化品位，进而也就提高了知名度。

coarse_types："组织机构","人","位置"

那么输出应该如下:
{ "entities": []（不存在任何符合分类集合中的名词，所以返回一个空的 entities 列表）}

