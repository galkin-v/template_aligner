import os
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv


class TemplateGenerator:
    """
    A class to generate product description templates from info files
    and save them as individual text files in an output folder.
    """

    MODEL_NAME = "gemini/gemini-2.5-flash"
    PRODUCT_INFO_FOLDER = Path("product_info")
    OUTPUT_FOLDER = Path("aligned_templates")
    ALIGNMENT_TEMPLATE_PATH = Path("templates/base_alignment_template.txt")
    GENERATION_TEMPLATE_PATH = Path("templates/base_generation_template.txt")
    TEMPLATE_ARGS = {}

    def __init__(self, product_folder=None, output_folder=None, model_name=None):
        """
        Initializes the TemplateGenerator.

        Args:
            product_folder (Path, optional): Path to the folder with product info files. Defaults to self.PRODUCT_INFO_FOLDER.
            output_folder (Path, optional): Path to the folder where templates will be saved. Defaults to self.OUTPUT_FOLDER.
            model_name (str, optional): The name of the LLM model to use. Defaults to self.MODEL_NAME.
        """
        self.product_folder = product_folder or self.PRODUCT_INFO_FOLDER
        self.output_folder = output_folder or self.OUTPUT_FOLDER
        self.model_name = model_name or self.MODEL_NAME

        self._setup_environment()
        self._prepare_folders()

    def _setup_environment(self):
        """Loads environment variables and checks for the required API key."""
        load_dotenv()
        if not os.getenv("GEMINI_API_KEY"):
            print(
                "❌ ERROR: GEMINI_API_KEY not found in environment variables or .env file."
            )
            print(
                "Please create a .env file and add your key: GEMINI_API_KEY='your-key-here'"
            )
            sys.exit(1)

    def _prepare_folders(self):
        """Ensures the output folder exists."""
        self.output_folder.mkdir(exist_ok=True)
        print(f"📂 Output will be saved to '{self.output_folder}'")

    def _load_prompt_templates(self) -> tuple[str, str]:
        """Loads the alignment and generation templates from their files."""
        try:
            alignment_template = self.ALIGNMENT_TEMPLATE_PATH.read_text(
                encoding="utf-8"
            )
            generation_template = self.GENERATION_TEMPLATE_PATH.read_text(
                encoding="utf-8"
            )
            return alignment_template, generation_template
        except FileNotFoundError as e:
            print(f"❌ ERROR: Template file not found: {e.filename}")
            print(
                "Please ensure your 'templates' folder and files are set up correctly."
            )
            sys.exit(1)

    def _generate_and_save_template(
        self, product_file: Path, alignment_template: str, generation_template: str
    ):
        """Generates and saves a single aligned template for a given product."""
        product_name = product_file.stem
        output_file = self.output_folder / f"{product_name}_template.txt"

        if output_file.exists():
            print(f"✅ Skipping '{product_name}', template already exists.")
            return

        print(f"✨ Generating template for '{product_name}'...")

        try:
            product_info = product_file.read_text(encoding="utf-8")

            final_prompt = alignment_template.format(
                base_generation_template=generation_template,
                product_info=product_info,
                product_name=product_name,
                **self.TEMPLATE_ARGS,
            )

            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.6,
            )

            generated_content = response.choices[0].message.content
            output_file.write_text(generated_content, encoding="utf-8")
            print(f"✅ Saved template: {output_file}")

        except Exception as e:
            print(f"❌ Failed to generate template for '{product_name}'. Error: {e}")

    def run(self):
        """
        Executes the main logic: finds product info files and generates
        a template for each one that doesn't already have one.
        """
        alignment_template, generation_template = self._load_prompt_templates()

        if not self.product_folder.exists():
            print(f"❌ ERROR: Product info folder not found at '{self.product_folder}'")
            sys.exit(1)

        product_files = list(self.product_folder.glob("*.txt"))
        if not product_files:
            print(f"⚠️  Warning: No '.txt' files found in '{self.product_folder}'.")
            return

        for product_file in product_files:
            self._generate_and_save_template(
                product_file, alignment_template, generation_template
            )

        print("\n🎉 All tasks complete.")


if __name__ == "__main__":
    generator = TemplateGenerator()
    generator.run()
