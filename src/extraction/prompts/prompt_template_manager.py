import json
import os
import importlib
from string import Template
from typing import Dict, List, Union, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {"system": "system", "user": "user", "assistant": "assistant"}
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)
        self.templates_dir = os.path.join(package_dir, "templates")
        self._load_templates()

    def _load_templates(self) -> None:
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory '{self.templates_dir}' does not exist.")
        logger.info(f"Loading templates from directory: {self.templates_dir}")

        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]
                module_name = f"src.extraction.prompts.templates.{script_name}"
                try:
                    module = importlib.import_module(module_name)
                    if not hasattr(module, "prompt_template"):
                        logger.info(f"Module '{module_name}' does not define a 'prompt_template', skipping.")
                        continue
                    prompt_template = module.prompt_template

                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                        isinstance(item, dict) and "role" in item and "content" in item for item in prompt_template
                    ):
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(item["role"], item["role"])
                            if not isinstance(item["content"], Template):
                                item["content"] = Template(item["content"])
                        self.templates[script_name] = prompt_template
                    else:
                        raise TypeError(f"Invalid prompt_template format in '{module_name}.py'.")

                except Exception as e:
                    logger.error(f"Failed to load template from '{module_name}.py': {e}")
                    raise

    def build_chat_prompt(
            self,
            template_name: str,
            new_passage: Union[str, Dict[str, Any]],
            few_shot: Optional[List[Tuple[Any, Any]]] = None,
            extra_vars: Optional[Dict[str, Any]] = None,
            max_few_shot: int = 3
    ) -> List[Dict[str, Any]]:
        """
        构建 chat-style prompt：
        1. system 模板
        2. few-shot 示例
        3. 当前 passage

        Args:
            template_name: 模板名称
            new_passage: 当前文本（字符串或 dict）
            few_shot: 动态 few-shot 示例 [(example_input, example_output)]
            extra_vars: 模板里额外占位符
            max_few_shot: 每条 prompt 最多 few-shot 数量

        Returns:
            List[Dict[str, Any]]: 可直接传给 Chat 模型的消息列表
        """
        extra_vars = extra_vars or {}
        # passage 保证是字符串
        passage_content = (
            new_passage if isinstance(new_passage, str)
            else json.dumps(new_passage, ensure_ascii=False)
        )
        extra_vars["passage"] = passage_content

        template = self.get_template(template_name)
        prompt_list: List[Dict[str, Any]] = []

        # --- 1. system 消息 ---
        if isinstance(template, list):
            for msg in template:
                content = msg["content"]
                if isinstance(content, dict):
                    content = json.dumps(content, ensure_ascii=False, indent=2)
                if isinstance(content, Template):
                    content = content.substitute(**extra_vars)
                prompt_list.append({"role": msg["role"], "content": content+'\n'})
        elif isinstance(template, Template):
            prompt_list.append({"role": "system", "content": template.substitute(**extra_vars)+'\n'})
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

        # --- 2. few-shot 示例插入 ---
        if few_shot:
            for example_in, example_out in few_shot[:max_few_shot]:
                if isinstance(example_in, dict):
                    example_in = json.dumps(example_in, ensure_ascii=False, indent=2)
                if isinstance(example_out, dict):
                    example_out = json.dumps(example_out, ensure_ascii=False, indent=2)
                prompt_list.append({"role": "user", "content": example_in})
                prompt_list.append({"role": "assistant", "content": example_out})

        # --- 3. 当前 passage ---
        if isinstance(new_passage, dict):
            passage_content = json.dumps(new_passage, ensure_ascii=False, indent=2)
        prompt_list.append({"role": "user", "content": passage_content+'\n'})

        return prompt_list

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found.")
        return self.templates[name]
