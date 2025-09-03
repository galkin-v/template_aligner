import os
import sqlite3
import sys
from pathlib import Path

import litellm
from dotenv import load_dotenv
from gigachat_litellm_client import gigachat_handler


class TemplateAlignerDb:
    """
    A class to generate and manage product description templates using an LLM.

    This class handles:
    - Setting up the environment and database.
    - Reading product information from files.
    - Generating new templates for products not already in the database.
    - Storing and retrieving templates from an SQLite database.
    """

    MODEL_NAME = "gemini/gemini-2.5-flash"
    PRODUCT_INFO_FOLDER = Path("product_info")
    ALIGNMENT_TEMPLATE_PATH = Path("templates/base_alignment_template.txt")
    GENERATION_TEMPLATE_PATH = Path("templates/base_generation_template.txt")
    DB_PATH = Path("templates.db")
    TEMPLATE_ARGS = {}

    def __init__(self, db_path=None, model_name=None):
        """
        Initializes the TemplateAligner.

        Args:
            db_path (Path, optional): Path to the SQLite database file. Defaults to self.DB_PATH.
            model_name (str, optional): The name of the LLM model to use. Defaults to self.MODEL_NAME.
        """
        self.db_path = db_path or self.DB_PATH
        self.model_name = model_name or self.MODEL_NAME

        if self.model_name in [
            "gigachat-provider/GigaChat-2",
            "gigachat-provider/GigaChat-2-Pro",
            "gigachat-provider/GigaChat-2-Max",
        ]:
            litellm.custom_provider_map = [
                {"provider": "gigachat-provider", "custom_handler": gigachat_handler}
            ]

        self.conn = None
        self.cursor = None

        self._setup_environment()
        self._setup_database()

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

    def _setup_database(self):
        """Connects to the SQLite database and creates the products table if it doesn't exist."""
        print(f"🗄️  Connecting to database at '{self.db_path}'...")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                template_content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()
        print("✅ Database setup complete.")

    def _get_existing_products(self) -> set:
        """Fetches the names of all products that already have a template in the database."""
        self.cursor.execute("SELECT name FROM products;")
        return {row[0] for row in self.cursor.fetchall()}

    def _save_template_to_db(self, product_name: str, template_content: str):
        """Saves a new product and its generated template to the database."""
        try:
            self.cursor.execute(
                "INSERT INTO products (name, template_content) VALUES (?, ?);",
                (product_name, template_content),
            )
            self.conn.commit()
            print(f"💾 Saved template for '{product_name}' to the database.")
        except sqlite3.IntegrityError:
            print(
                f"⚠️  Warning: Product '{product_name}' already exists in the database."
            )

    def _generate_template_for_product(
        self,
        product_info: str,
        product_name: str,
        alignment_template: str,
        generation_template: str,
    ) -> str | None:
        """Generates an aligned template for a given product using the LLM."""
        print(f"✨ Generating template for new product: {product_name}...")
        try:
            final_prompt = alignment_template.format(
                base_generation_template=generation_template,
                product_info=product_info,
                product_name=product_name,
                **self.TEMPLATE_ARGS,
            )

            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": final_prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Failed to generate template for {product_name}. Error: {e}")
            return None

    def run(self):
        """
        Executes the main logic: finds new products, generates templates, and saves them.
        """
        existing_products = self._get_existing_products()
        if existing_products:
            print(
                f"ℹ️  Found {len(existing_products)} products with existing templates in the database."
            )
        else:
            print(
                "ℹ️  Database is empty. Will generate templates for all found products."
            )

        try:
            alignment_template = self.ALIGNMENT_TEMPLATE_PATH.read_text(
                encoding="utf-8"
            )
            generation_template = self.GENERATION_TEMPLATE_PATH.read_text(
                encoding="utf-8"
            )
        except FileNotFoundError as e:
            print(f"❌ ERROR: Template file not found: {e.filename}", file=sys.stderr)
            sys.exit(1)

        if not self.PRODUCT_INFO_FOLDER.is_dir():
            print(
                f"❌ ERROR: Product info folder not found at '{self.PRODUCT_INFO_FOLDER}'",
                file=sys.stderr,
            )
            sys.exit(1)

        product_files = list(self.PRODUCT_INFO_FOLDER.glob("*.txt"))
        if not product_files:
            print(f"⚠️  Warning: No '.txt' files found in '{self.PRODUCT_INFO_FOLDER}'.")
            return

        new_templates_generated = 0
        for product_file in product_files:
            product_name = product_file.stem
            if product_name in existing_products:
                print(f"✅ Skipping '{product_name}', template already in database.")
                continue

            product_info = product_file.read_text(encoding="utf-8")
            generated_content = self._generate_template_for_product(
                product_info, product_name, alignment_template, generation_template
            )

            if generated_content:
                self._save_template_to_db(product_name, generated_content)
                new_templates_generated += 1

        print(
            f"\n🎉 All tasks complete. Generated {new_templates_generated} new templates."
        )

    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("🗄️  Database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def view_templates_from_db(db_path: Path):
        """
        Fetches and prints all product templates from the database.
        This is a static method and can be called without creating an instance.
        """
        if not db_path.exists():
            print(f"❌ Database file '{db_path}' does not exist.")
            return

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, template_content, created_at FROM products ORDER BY created_at;"
        )
        rows = cursor.fetchall()

        if not rows:
            print("ℹ️  No templates found in the database.")
        else:
            print(f"📦 Found {len(rows)} templates in the database:\n")
            for name, content, created_at in rows:
                print(f"--- {name} (created at {created_at}) ---")
                print(content[:500] + ("..." if len(content) > 500 else ""))
                print()
        conn.close()


if __name__ == "__main__":
    with TemplateAlignerDb(model_name="gigachat-provider/GigaChat-2") as aligner:
        aligner.run()

    print("\n--- Viewing existing templates ---")
    TemplateAlignerDb.view_templates_from_db(TemplateAlignerDb.DB_PATH)
