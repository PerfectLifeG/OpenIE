
# 更新时间
## 2025-11-10
- [x] 调整自我验证的 prompt，只需要 answer
- [x] 调整提取粗粒度-实体的 prompt，要求必须是 named entity，将 coarse_type 转移到 user_prompt 中的结果
- 输出结果稳定了。

## 2025-11-08
- [x] 增加对粗粒度的自我验证类。
    - [x] 对单独样本进行 prompt 构建
    - [x] 对单独样本进行验证
    - [x] 读取 json 文件中每个样本进行以上操作，结果输出为新的 json 文件。
- [x] 增加生成式提取实体-粗粒度的方法。
    - [x] 对每个样例生成一组抽取实体和粗粒度的 prompt，并行发送，接收所有回答并整合到同一 id 答案下，如果本身有 id ，则沿用，如果没有，则自己根据样例顺序创建
    - [x] 补全 llm 生成返回带实体标记的代码


## 2025-10-22
更新了 prompt，提升了模型效果。

## 2025-10-19
增加了`requirements.txt`，方便环境配置。
运行代码时，注意安装`requirements.txt`中的依赖包。

# 运行
1. 使用 Anaconda 虚拟环境`conda create -y -n ccfner python=3.10 && conda activate ccfner`
2. 在本地配置好 ollama，下载`configs/default.yaml`中指定的模型，直接运行`test.ipynb`中的代码即可。默认运行对`dev1.json`的评估测试。

## 模型
可以通过修改`configs/default.yaml`中的`model_name`来切换不同的模型。  
修改`configs/default.yaml`中的`llm`配置，可以切换不同的 LLM 架构。

# 描述
这是一个最简单的版本，整个运行流程如下：  
从数据集中抽取数据，对每份数据提取出`sentence`和`coarse_types`，添加基本`prompt`，传给`llm`，让`llm`一次性生成指定结构的`json`结构数据传回，最后对传回的数据与数据集中的标准答案进行比较，得到评估结果。

## 代码阅读
从`src/main.py`开始逐行阅读，极简代码。

