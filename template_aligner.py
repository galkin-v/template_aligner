import os
from pathlib import Path

import google.generativeai as genai


class ProductTemplateAligner:
    """
    A class to generate customized product templates by aligning product information
    with base templates and a generative AI model.

    This class provides a user-friendly API to automate content creation by dynamically
    compiling prompts with user-defined arguments.

    Attributes:
        model (genai.GenerativeModel): The generative AI model instance.
        aligned_templates_folder (str): Path to the folder for storing output templates.
        product_info_folder (str): Path to the folder containing product information files.
        templates_folder (str): Path to the folder containing base template files.
        prompt_args (dict): A dictionary of custom arguments for prompt formatting.
    """

    def __init__(
        self,
        model_name: str,
        aligned_templates_folder="aligned_templates",
        product_info_folder="product_info",
        templates_folder="templates",
        **kwargs,
    ):
        """
        Initializes the ProductTemplateAligner.

        Args:
            model_name (str): The name of the generative model to use (e.g., 'gemini-pro').
            **kwargs: Arbitrary keyword arguments that will be available to format the
                      prompt templates. For example, tone="formal", audience="experts".
        """
        self.aligned_templates_folder = aligned_templates_folder
        self.product_info_folder = product_info_folder
        self.templates_folder = templates_folder

        self.prompt_args = kwargs

        self.model = genai.GenerativeModel(model_name)

        self._alignment_template = None
        self._generation_template = None

    def _load_variables(self):
        """
        Private method to create directories and load base templates from files.
        """
        os.makedirs(self.aligned_templates_folder, exist_ok=True)
        os.makedirs(self.product_info_folder, exist_ok=True)
        os.makedirs(self.templates_folder, exist_ok=True)

        if not os.listdir(self.product_info_folder):
            raise FileNotFoundError(
                f"Product information folder is empty! Load your documents into the folder '{self.product_info_folder}' and run the script again."
            )

        alignment_template_path = os.path.join(self.templates_folder, "base_alignment_template.txt")
        generation_template_path = os.path.join(self.templates_folder, "base_generation_template.txt")

        if not os.path.exists(alignment_template_path) or not os.path.exists(generation_template_path):
            raise FileNotFoundError(
                f"Base templates are missing. Ensure 'base_alignment_template.txt' and 'base_generation_template.txt' are in the '{self.templates_folder}' folder."
            )

        with open(alignment_template_path, "r") as f:
            self._alignment_template = f.read()
        with open(generation_template_path, "r") as f:
            self._generation_template = f.read()
        print("Base templates loaded successfully.")

    def _align_template_to_product(self, product_info_content: str, product_name: str) -> str:
        """
        Private method to format the prompts and generate content using the model.

        Args:
            product_info_content (str): The content from a product information file.

        Returns:
            str: The generated content from the AI model.
        """

        final_prompt = self._alignment_template.format(
            base_generation_template=self._generation_template,
            product_info=product_info_content,
            product_name=product_name,
            **self.prompt_args,
        )

        print("Generating content with the final compiled prompt...")
        response = self.model.generate_content(final_prompt)
        return response.text

    def run(self):
        """
        The main public method to run the entire template generation process.

        This method loads all necessary files and then iterates through the product
        information to create aligned templates.
        """
        self._load_variables()

        for product_filename in os.listdir(self.product_info_folder):
            product_info_full_path = os.path.join(self.product_info_folder, product_filename)
            base, _ = os.path.splitext(product_filename)
            template_filename = f"{base}_template.txt"
            template_full_path = os.path.join(self.aligned_templates_folder, template_filename)

            if not os.path.exists(template_full_path):
                print(f"\nProcessing file: {product_filename}...")
                with open(product_info_full_path, "r") as product_info_file:
                    product_content = product_info_file.read()

                    generated_content = self._align_template_to_product(
                        product_content, product_name=Path(product_info_full_path).stem
                    )

                    with open(template_full_path, "w") as f:
                        f.write(generated_content)
                    print(f"Successfully created template: {template_filename}")
            else:
                print(f"\nTemplate for {product_filename} already exists. Skipping.")
                print(f"\nTemplate for {product_filename} already exists. Skipping.")
