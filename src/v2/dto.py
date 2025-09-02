from typing import Any

from pydantic import BaseModel, Field


class Serializable(BaseModel):
    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        # Получаем стандартную схему
        schema = super().model_json_schema(*args, **kwargs)
        # Удаляем заголовок из корневого элемента
        schema.pop("title", None)
        # Удаляем заголовки из всех свойств
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema


class ProductInfo(Serializable):
    name: str
    aliases: list[str]
    product_description: str
    extraction_rules: str


class ProductFewShot(Serializable):
    description: str
    dialog_history: str
    current_utterance: str
    analysis: str
    result: dict[str, list[str]]
    is_key_example: bool = Field(
        default=False, description="Помечает этот пример как основной для показа в мульти-продуктовых промптах."
    )


class FilledProduct(Serializable):
    info: ProductInfo
    few_shots: list[ProductFewShot]

    def get_key_example(self) -> list[ProductFewShot]:
        """Возвращает ключевой пример или первый, если ключевой не задан."""
        exmps = [ex for ex in self.few_shots if ex.is_key_example]
        return exmps or []

    def compile_description_block(self) -> str:
        """Компилирует блок с описанием и правилами продукта."""
        return f"""\
- **"{self.info.name}"**
	- **Описание:** {self.info.product_description}
	- **Ключевые понятия, на которые стоит ориентироваться, включают (но не ограничиваются ими):** {", ".join(self.info.aliases)}
	- **Правила извлечения:**
{self.info.extraction_rules}"""


class FactCheckerProducts(BaseModel):
    products: list[FilledProduct]
    cross_product_example: list[ProductFewShot] | None = Field(
        default=None,
        description="Кросс-продуктовый пример, который учит модель обрабатывать несколько продуктов в одном ответе.",
    )

    def compile_product_specific_section(self) -> str:
        """
        Собирает единый блок инструкций для всех переданных продуктов,
        включая индивидуальные и кросс-продуктовые примеры.
        """
        # 1. Собираем описания и правила для всех продуктов
        product_descriptions = [p.compile_description_block() for p in self.products]
        product_list_str = "\n\n".join(product_descriptions)

        full_product_section = f"""\
### СПИСОК ПРОДУКТОВ ДЛЯ АНАЛИЗА И ИХ ПРАВИЛА

{product_list_str}
"""

        return f"{full_product_section}"

    def compile_few_shots(self) -> str:
        examples_str_list = []
        example_counter = 1

        # Добавляем по одному ключевому примеру на каждый продукт
        for product in self.products:
            key_examples = product.get_key_example() if len(self.products) > 1 else product.few_shots
            for key_example in key_examples:
                str_resp = (
                    json.dumps(key_example.result, ensure_ascii=False, indent=2).replace("{", "{{").replace("}", "}}")
                )
                example_str = f"""\
**Пример {example_counter} (для продукта "{product.name}"): {key_example.description}**
`ИСТОРИЯ_ДИАЛОГА`: {key_example.dialog_history}
`ТЕКУЩАЯ_РЕПЛИКА`: {key_example.current_utterance}
**Анализ:** {key_example.analysis}
**Результат:**
`{str_resp}`
"""
                examples_str_list.append(example_str)
                example_counter += 1

        # 3. Добавляем кросс-продуктовый пример, если он есть и продуктов больше одного
        if len(self.products) > 1 and self.cross_product_example:
            for ex in self.cross_product_example:
                # ex = self.cross_product_example
                str_resp = json.dumps(ex.result, ensure_ascii=False, indent=2).replace("{", "{{").replace("}", "}}")

                cross_example_str = f"""\
**Пример {example_counter} (кросс-продуктовый): {ex.description}**
`ИСТОРИЯ_ДИАЛОГА`: {ex.dialog_history}
`ТЕКУЩАЯ_РЕПЛИКА`: {ex.current_utterance}
**Анализ:** {ex.analysis}
**Результат:**
`{str_resp}`
"""
                examples_str_list.append(cross_example_str)

        exemples = "\n".join(examples_str_list)

        return exemples


if __name__ == "__main__":
    import json

    print(json.dumps(ProductInfo.model_json_schema(by_alias=False), indent=4, ensure_ascii=False))
