#!/usr/bin/env python3
"""CLI entry point for local-photo-search."""

import json
import os

import click

from photosearch.db import PhotoDB
from photosearch.index import index_directory
from photosearch.search import search_semantic, search_by_color, search_by_place, search_combined, symlink_results, make_results_subdir


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Local photo search — find photos by person, place, description, or color."""
    pass


@cli.command()
@click.argument("photo_dir", type=click.Path(exists=True))
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
@click.option("--batch-size", default=8, help="Batch size for CLIP embedding.")
@click.option("--no-clip", is_flag=True, help="Skip CLIP embedding generation.")
@click.option("--no-colors", is_flag=True, help="Skip dominant color extraction.")
def index(photo_dir, db, batch_size, no_clip, no_colors):
    """Index a directory of photos."""
    index_directory(
        photo_dir=photo_dir,
        db_path=db,
        batch_size=batch_size,
        enable_clip=not no_clip,
        enable_colors=not no_colors,
    )


@cli.command()
@click.option("--query", "-q", help="Semantic search query (natural language).")
@click.option("--person", "-p", help="Search by person name.")
@click.option("--place", help="Search by place name.")
@click.option("--color", "-c", help="Search by dominant color (name or #hex).")
@click.option("--face", type=click.Path(exists=True), help="Search by reference face image.")
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
@click.option("--limit", "-n", default=10, help="Max number of results.")
@click.option("--symlink-dir", default="results", help="Directory to symlink matching photos into.")
@click.option("--no-symlinks", is_flag=True, help="Don't create symlinked results folder.")
@click.option("--json-output", is_flag=True, help="Output results as JSON.")
def search(query, person, place, color, face, db, limit, symlink_dir, no_symlinks, json_output):
    """Search indexed photos."""
    if not any([query, person, place, color, face]):
        click.echo("Please provide at least one search criterion. See --help for options.")
        return

    with PhotoDB(db) as photo_db:
        # TODO: person and face search will be added in M2
        if person:
            click.echo("Person search will be available after Milestone 2.")
            return
        if face:
            click.echo("Face search will be available after Milestone 2.")
            return

        results = search_combined(
            db=photo_db,
            query=query,
            color=color,
            place=place,
            limit=limit,
        )

        if not results:
            click.echo("No matching photos found.")
            return

        # Display results
        if json_output:
            # Clean up for JSON serialization
            for r in results:
                for k in list(r.keys()):
                    if r[k] is None:
                        del r[k]
            click.echo(json.dumps(results, indent=2))
        else:
            click.echo(f"\nFound {len(results)} matching photos:\n")
            click.echo(f"{'#':<4} {'Filename':<20} {'Date':<22} {'Score':<8} {'Colors'}")
            click.echo("-" * 80)
            for i, r in enumerate(results, 1):
                filename = r.get("filename", "?")
                date = r.get("date_taken", "")[:19] if r.get("date_taken") else ""
                score = f"{r.get('score', 0):.3f}" if "score" in r else ""
                colors = ""
                if r.get("dominant_colors"):
                    try:
                        color_list = json.loads(r["dominant_colors"])
                        colors = " ".join(color_list[:3])
                    except (json.JSONDecodeError, TypeError):
                        pass
                click.echo(f"{i:<4} {filename:<20} {date:<22} {score:<8} {colors}")

        # Create symlinked results subfolder named by timestamp + query
        if not no_symlinks and results:
            subfolder = make_results_subdir(
                symlink_dir,
                {"query": query, "color": color, "place": place, "person": person},
            )
            result_path = symlink_results(results, output_dir=subfolder)
            click.echo(f"\nResults symlinked to: {result_path}")


@cli.command()
@click.option("--db", default="photo_index.db", help="Path to the SQLite database file.")
def stats(db):
    """Show database statistics."""
    if not os.path.exists(db):
        click.echo(f"Database not found: {db}")
        return

    with PhotoDB(db) as photo_db:
        count = photo_db.photo_count()
        click.echo(f"Database: {db}")
        click.echo(f"Total indexed photos: {count}")

        # Show sample entries
        if count > 0:
            rows = photo_db.conn.execute(
                "SELECT filename, date_taken, camera_model, place_name FROM photos LIMIT 5"
            ).fetchall()
            click.echo(f"\nSample entries:")
            for row in rows:
                click.echo(f"  {row['filename']}  {row['date_taken'] or ''}  {row['camera_model'] or ''}")


if __name__ == "__main__":
    cli()
