import asyncio
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Literal, Optional, Any, Type, get_type_hints

from agents import (
    Agent,
    Runner,
    OpenAIProvider,
    RunConfig,
    TResponseInputItem,
    trace,
    RunResult,
    AgentOutputSchemaBase,
)
from openai import AsyncOpenAI
import json


# 定义数据类
@dataclass
class UserStory:
    id: str
    description: str


@dataclass
class BasicFlowStep:
    order: str
    actor: str
    action: str


@dataclass
class UseCaseDescription:
    num: str
    description: str
    basic_flow: List[BasicFlowStep]


@dataclass
class Actor:
    name: str


@dataclass
class UseCase:
    name: str
    includes: List[str]
    extends: List[str]


@dataclass
class Message:
    sender: str
    receiver: str
    message_type: str
    order: str


@dataclass
class Attribute:
    name: str
    type: str


@dataclass
class Method:
    name: str
    return_type: str


@dataclass
class Association:
    target_class: str
    association_type: str


@dataclass
class ClassInfo:
    name: str
    attributes: List[Attribute]
    methods: List[Method]
    associations: List[Association]


@dataclass
class UserStories:
    user_stories: List[UserStory]


@dataclass
class UseCaseDescriptions:
    use_case_descriptions: List[UseCaseDescription]


@dataclass
class UseCaseDiagram:
    use_case_diagram: Dict[str, List[Actor]]


@dataclass
class SequenceDiagrams:
    sequence_diagrams: List[Dict[str, List[Message]]]


@dataclass
class ClassDiagrams:
    class_diagrams: List[ClassInfo]


@dataclass
class UMLModelOutput:
    user_stories: UserStories
    use_case_descriptions: UseCaseDescriptions
    use_case_diagram: UseCaseDiagram
    sequence_diagrams: SequenceDiagrams
    class_diagrams: ClassDiagrams


@dataclass
class EvaluationFeedback:
    score: Literal["pass", "needs_improvement", "fail"]
    feedback: str


# 定义UML模型生成Agent
class UMLModelGeneratorOutputSchema(AgentOutputSchemaBase):
    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return "UMLModelGeneratorOutputSchema"

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_stories": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }},
                "use_case_descriptions": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "num": {"type": "string"},
                        "description": {"type": "string"},
                        "basic_flow": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "order": {"type": "string"},
                                "actor": {"type": "string"},
                                "action": {"type": "string"}
                            }
                        }}
                    }
                }},
                "use_case_diagram": {"type": "object"},
                "sequence_diagrams": {"type": "array", "items": {"type": "object"}},
                "class_diagrams": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "attributes": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"}
                            }
                        }},
                        "methods": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "return_type": {"type": "string"}
                            }
                        }},
                        "associations": {"type": "array", "items": {
                            "type": "object",
                            "properties": {
                                "target_class": {"type": "string"},
                                "association_type": {"type": "string"}
                            }
                        }}
                    }
                }}
            },
            "required": ["user_stories", "use_case_descriptions", "use_case_diagram", "sequence_diagrams",
                         "class_diagrams"]
        }

    def is_strict_json_schema(self) -> bool:
        return False

    def validate_json(self, json_str: str) -> Any:
        json_obj = json.loads(json_str)

        # 递归地将字典转换为对象
        def dict_to_dataclass(target_cls: Type, data: Any) -> Any:
            if isinstance(data, dict):
                # 获取目标类的所有字段
                field_types = get_type_hints(target_cls)
                # 获取目标类的构造函数参数
                init_fields = [f.name for f in fields(target_cls)]

                # 递归转换每个字段
                converted_data = {}
                for key, value in data.items():
                    if key in field_types and key in init_fields:
                        field_type = field_types[key]
                        if hasattr(field_type, "__origin__") and field_type.__origin__ == list:
                            # 处理列表类型
                            item_type = field_type.__args__[0]
                            converted_data[key] = [dict_to_dataclass(item_type, item) for item in value]
                        elif hasattr(field_type, "__origin__") and field_type.__origin__ == dict:
                            # 处理字典类型
                            key_type, value_type = field_type.__args__
                            converted_data[key] = {k: dict_to_dataclass(value_type, v) for k, v in value.items()}
                        else:
                            # 处理普通类型
                            converted_data[key] = dict_to_dataclass(field_type, value)
                    elif key in init_fields:
                        converted_data[key] = value

                # 创建目标类的实例
                init_args = {}
                for field in init_fields:
                    init_args[field] = converted_data.get(field, None)

                return target_cls(**init_args)
            elif isinstance(data, list):
                # 处理列表
                return [dict_to_dataclass(target_cls.__args__[0], item) for item in data]
            else:
                # 基本类型直接返回
                return data

        # 转换用户故事
        user_stories = UserStories(
            user_stories=[dict_to_dataclass(UserStory, story) for story in json_obj.get("user_stories", [])]
        )

        # 转换用例描述
        use_case_descriptions = UseCaseDescriptions(
            use_case_descriptions=[]
        )
        for use_case in json_obj.get("use_case_descriptions", []):
            if isinstance(use_case, dict):
                use_case_desc = dict_to_dataclass(UseCaseDescription, {
                    "num": use_case.get("num", ""),
                    "description": use_case.get("description", ""),
                    "basic_flow": [dict_to_dataclass(BasicFlowStep, step) for step in use_case.get("basic_flow", [])]
                })
                use_case_descriptions.use_case_descriptions.append(use_case_desc)

        # 转换用例图
        use_case_diagram = UseCaseDiagram(
            use_case_diagram={
                k: [dict_to_dataclass(Actor, {attr: v[attr] if attr in v else None for attr in ['name']})
                    for v in (values if isinstance(values, list) else [])]
                for k, values in (json_obj.get("use_case_diagram", {}).items()
                                  if isinstance(json_obj.get("use_case_diagram", {}), dict) else {})
            }
        )

        # 转换序列图
        sequence_diagrams = SequenceDiagrams(
            sequence_diagrams=[
                {k: [dict_to_dataclass(Message, msg) for msg in v]
                 for k, v in (diagram.items() if isinstance(diagram, dict) else {})}
                for diagram in json_obj.get("sequence_diagrams", [])
            ]
        )

        # 转换类图
        class_diagrams = ClassDiagrams(
            class_diagrams=[
                dict_to_dataclass(ClassInfo, {
                    "name": cls.get("name", ""),
                    "attributes": [dict_to_dataclass(Attribute, attr) for attr in cls.get("attributes", [])],
                    "methods": [dict_to_dataclass(Method, method) for method in cls.get("methods", [])],
                    "associations": [dict_to_dataclass(Association, assoc) for assoc in cls.get("associations", [])]
                }) for cls in
                (json_obj.get("class_diagrams", []) if isinstance(json_obj.get("class_diagrams", []), list) else [])
            ]
        )

        return UMLModelOutput(
            user_stories=user_stories,
            use_case_descriptions=use_case_descriptions,
            use_case_diagram=use_case_diagram,
            sequence_diagrams=sequence_diagrams,
            class_diagrams=class_diagrams
        )


# 配置OpenAIProvider
provider = OpenAIProvider(
    openai_client=AsyncOpenAI(
        base_url="",
        api_key="",
    ),
    use_responses=False,
)


async def main() -> None:
    msg: str = ("我想要对一个AI全景系统建模，用户主要有全景图片采集者、AI维护者、管理者和用户，功能是用户登录网站查看全景图片并且可以向AI提问,"
                "全景图片采集者可以采集全景图片并放入网站数据，AI维护者部署可交互AI，管理者管理其他用户")
    input_items: List[TResponseInputItem] = [{"content": msg, "role": "user"}]
    uml_model: Optional[UMLModelOutput] = None

    with trace("LLM as a judge"):
        while True:
            # 生成UML模型
            uml_model_generator = Agent(
                name="uml_model_generator",
                model="gpt-4o",
                instructions=(
                    "Generate a UML model based on the user's input. "
                    "Include use case diagram, sequence diagrams, and class diagrams."
                ),
                output_type=UMLModelGeneratorOutputSchema
            )
            uml_model_generator.output_type = UMLModelGeneratorOutputSchema()
            uml_model_result: RunResult = await Runner.run(
                uml_model_generator,
                input_items,
                run_config=RunConfig(model_provider=provider)
            )

            # 将结果转换为字典后序列化为JSON
            uml_model_dict = asdict(uml_model_result.final_output)
            input_items = [{"content": json.dumps(uml_model_dict), "role": "assistant"}]
            uml_model = uml_model_result.final_output

            # 评估UML模型
            evaluator = Agent(
                name="evaluator",
                model="gpt-4o",
                instructions=(
                    "You evaluate a UML model and decide if it's good enough. "
                    "If it's not good enough, you provide feedback on what needs to be improved. "
                    "Never give it a pass on the first try."
                ),
                output_type=EvaluationFeedback
            )
            evaluator_result: RunResult = await Runner.run(
                evaluator, input_items, run_config=RunConfig(model_provider=provider)
            )
            result: EvaluationFeedback = evaluator_result.final_output

            print(f"Evaluator score: {result.score}")
            if result.score == "pass":
                print("UML model is good enough, exiting.")
                break
            print(result.feedback)
            print("Re-running with feedback")
            input_items.append({"content": f"Feedback: {result.feedback}", "role": "user"})

    if uml_model:
        print("Final UML Model Output:")
        # 打印用户故事
        print("User Stories:")
        for story in uml_model.user_stories.user_stories:
            print(f"ID: {story.id}, Description: {story.description}")
        # 打印用例描述
        print("\nUse Case Descriptions:")
        for use_case_desc in uml_model.use_case_descriptions.use_case_descriptions:
            print(f"NUM: {use_case_desc.num}, Description: {use_case_desc.description}")
            print("Basic Flow:")
            for step in use_case_desc.basic_flow:
                print(f"  {step.order}. {step.actor} {step.action}")


if __name__ == "__main__":
    asyncio.run(main())