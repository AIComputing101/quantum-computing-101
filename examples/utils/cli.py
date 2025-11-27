#!/usr/bin/env python3
"""
Quantum Computing 101 - Command Line Interface

A simple CLI to help users navigate and run examples.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def list_modules():
    """List all available modules."""
    examples_dir = Path(__file__).parent.parent
    modules = []

    for module_dir in sorted(examples_dir.glob("module*")):
        if module_dir.is_dir():
            examples = list(module_dir.glob("*.py"))
            modules.append(
                {"name": module_dir.name, "path": module_dir, "examples": len(examples)}
            )

    return modules


def list_examples(module_name):
    """List examples in a specific module."""
    examples_dir = Path(__file__).parent.parent
    module_path = examples_dir / module_name

    if not module_path.exists():
        return []

    examples = []
    for example_file in sorted(module_path.glob("*.py")):
        examples.append({"name": example_file.name, "path": example_file})

    return examples


def run_example(module_name, example_name, args=None):
    """Run a specific example."""
    examples_dir = Path(__file__).parent.parent
    example_path = examples_dir / module_name / example_name

    if not example_path.exists():
        print(f"Error: Example {example_path} not found")
        return 1

    # Build command
    cmd = [sys.executable, str(example_path)]
    if args:
        cmd.extend(args)

    # Run the example
    try:
        result = subprocess.run(cmd, cwd=examples_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error running example: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Computing 101 - Educational Platform CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quantum101 list                           # List all modules
  quantum101 list module1_fundamentals      # List examples in module 1
  quantum101 run module1_fundamentals 01_classical_vs_quantum_bits.py
  quantum101 run module4_algorithms 02_grovers_search_algorithm.py --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List modules or examples")
    list_parser.add_argument(
        "module", nargs="?", help="Module name to list examples from"
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an example")
    run_parser.add_argument("module", help="Module name")
    run_parser.add_argument("example", help="Example filename")
    run_parser.add_argument("args", nargs="*", help="Arguments to pass to the example")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show project information")

    args = parser.parse_args()

    if args.command == "list":
        if args.module:
            # List examples in specific module
            examples = list_examples(args.module)
            if examples:
                print(f"\n📚 Examples in {args.module}:")
                for i, example in enumerate(examples, 1):
                    print(f"  {i}. {example['name']}")
            else:
                print(f"❌ Module '{args.module}' not found")
                return 1
        else:
            # List all modules
            modules = list_modules()
            print("\n🚀 Quantum Computing 101 - Available Modules:")
            print("=" * 50)
            for module in modules:
                print(f"📂 {module['name']:<25} ({module['examples']} examples)")

            print(
                f"\n📊 Total: {len(modules)} modules, {sum(m['examples'] for m in modules)} examples"
            )
            print(
                "\nUse 'quantum101 list <module_name>' to see examples in a specific module"
            )

    elif args.command == "run":
        return run_example(args.module, args.example, args.args)

    elif args.command == "info":
        print("\n🚀 Quantum Computing 101")
        print("=" * 30)
        print("📚 A comprehensive quantum computing education platform")
        print("🎯 45 production-ready examples across 8 modules")
        print("📊 24,547+ lines of educational quantum computing code")
        print("🌍 Open source and community-driven")
        print("\n🔗 More info: https://github.com/AIComputing101/quantum-computing-101")

        # Show quick stats
        modules = list_modules()
        print(f"\n📈 Current Status:")
        print(f"   Modules: {len(modules)}")
        print(f"   Examples: {sum(m['examples'] for m in modules)}")
        print(f"   Lines of Code: 24,547+")

    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
