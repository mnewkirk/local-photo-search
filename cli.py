#!/usr/bin/env python3
"""CLI entry point for local-photo-search."""

import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Local photo search — find photos by person, place, description, or color."""
    pass


@cli.command()
@click.argument("photo_dir", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
def index(photo_dir, db):
    """Index a directory of photos."""
    click.echo(f"Indexing {photo_dir} → {db}")
    # TODO: M1 implementation


@cli.command()
@click.option("--query", "-q", help="Semantic search query (natural language).")
@click.option("--person", "-p", help="Search by person name.")
@click.option("--place", help="Search by place name.")
@click.option("--color", "-c", help="Search by dominant color.")
@click.option("--face", type=click.Path(exists=True), help="Search by reference face image.")
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
@click.option("--limit", "-n", default=10, help="Max number of results.")
@click.option("--symlink-dir", default="results", help="Directory to symlink matching photos into.")
def search(query, person, place, color, face, db, limit, symlink_dir):
    """Search indexed photos."""
    click.echo("Search not yet implemented.")
    # TODO: M4 implementation


if __name__ == "__main__":
    cli()
