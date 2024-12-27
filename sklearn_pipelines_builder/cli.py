"""Console script for sklearn_pipelines_builder."""
import sklearn_pipelines_builder

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for sklearn_pipelines_builder."""
    console.print("Replace this message by putting your code into "
               "sklearn_pipelines_builder.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
