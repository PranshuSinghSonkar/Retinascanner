"""Flask application entry point for RetinaAI."""

from flask import Flask


def create_app() -> Flask:
    """Create and configure the RetinaAI Flask application."""
    app = Flask(__name__)
    app.config.from_object("config.Config")

    @app.get("/")
    def index() -> str:
        return "RetinaAI is ready."

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
