#!/usr/bin/env python3
"""
Quantum Computing 101 - Examples Verification Tool

This script verifies the integrity and syntax of all example files without
executing them. Useful for quick validation after updates or before commits.

Usage:
    python verify_examples.py                    # Verify all examples
    python verify_examples.py --module module1   # Verify specific module
    python verify_examples.py --quick            # Quick syntax check only
"""

import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import ast


class ExampleVerifier:
    """Verify quantum computing examples for syntax and basic structure."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.root_dir = Path(__file__).parent
        self.examples_dir = self.root_dir / "examples"
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            prefix = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}
            print(f"{prefix.get(level, 'ℹ️')} {message}")

    def verify_syntax(self, filepath: Path) -> bool:
        """Check Python file for syntax errors."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, filepath.name, 'exec')
            return True
        except SyntaxError as e:
            self.errors.append(f"{filepath.name}: Syntax error at line {e.lineno}: {e.msg}")
            self.log(f"{filepath.name}: Syntax error at line {e.lineno}", "ERROR")
            return False
        except Exception as e:
            self.errors.append(f"{filepath.name}: {str(e)}")
            self.log(f"{filepath.name}: {str(e)}", "ERROR")
            return False

    def verify_structure(self, filepath: Path) -> bool:
        """Verify example has expected structure (main function, docstring, etc)."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            # Check for module docstring
            has_docstring = ast.get_docstring(tree) is not None
            if not has_docstring:
                self.warnings.append(f"{filepath.name}: Missing module docstring")
                if self.verbose:
                    self.log(f"{filepath.name}: Missing module docstring", "WARNING")
            
            # Check for main function
            has_main = any(
                isinstance(node, ast.FunctionDef) and node.name == 'main'
                for node in ast.walk(tree)
            )
            if not has_main:
                self.warnings.append(f"{filepath.name}: Missing main() function")
                if self.verbose:
                    self.log(f"{filepath.name}: Missing main() function", "WARNING")
            
            # Check for shebang
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            if not first_line.startswith('#!'):
                self.warnings.append(f"{filepath.name}: Missing shebang line")
                if self.verbose:
                    self.log(f"{filepath.name}: Missing shebang", "WARNING")
            
            return True
        except Exception as e:
            self.log(f"{filepath.name}: Structure check failed: {e}", "ERROR")
            return False

    def verify_imports(self, filepath: Path) -> bool:
        """Check for required imports (qiskit, matplotlib backend)."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            tree = ast.parse(code)
            
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or '')
            
            # Check for qiskit import
            has_qiskit = any('qiskit' in imp for imp in imports)
            
            # Check for matplotlib Agg backend (headless compatibility)
            has_matplotlib_backend = 'matplotlib.use' in code and 'Agg' in code
            
            if not has_qiskit:
                self.warnings.append(f"{filepath.name}: No qiskit import found")
            
            if 'matplotlib' in ' '.join(imports) and not has_matplotlib_backend:
                self.warnings.append(
                    f"{filepath.name}: Uses matplotlib but missing 'matplotlib.use('Agg')' for headless support"
                )
            
            return True
        except Exception as e:
            self.log(f"{filepath.name}: Import check failed: {e}", "ERROR")
            return False

    def verify_module(self, module_name: str, quick: bool = False) -> Tuple[int, int]:
        """Verify all examples in a module."""
        module_path = self.examples_dir / module_name
        
        if not module_path.exists():
            self.log(f"Module directory not found: {module_name}", "ERROR")
            return 0, 0
        
        print(f"\n📚 Verifying {module_name}...")
        
        example_files = sorted(module_path.glob("*.py"))
        example_files = [f for f in example_files if not f.name.startswith('__')]
        
        passed = 0
        failed = 0
        
        for example_file in example_files:
            # Always check syntax
            syntax_ok = self.verify_syntax(example_file)
            
            if not quick and syntax_ok:
                # Additional checks in non-quick mode
                self.verify_structure(example_file)
                self.verify_imports(example_file)
            
            if syntax_ok:
                print(f"  ✅ {example_file.name}")
                passed += 1
            else:
                print(f"  ❌ {example_file.name}")
                failed += 1
        
        return passed, failed

    def verify_all(self, module_filter: str = None, quick: bool = False) -> int:
        """Verify all examples or filtered by module."""
        print("🚀 Quantum Computing 101 - Example Verification")
        print("=" * 60)
        
        module_dirs = sorted(self.examples_dir.glob("module*"))
        module_dirs = [d for d in module_dirs if d.is_dir()]
        
        if module_filter:
            module_dirs = [d for d in module_dirs if module_filter in d.name]
            if not module_dirs:
                print(f"❌ No modules found matching: {module_filter}")
                return 1
        
        total_passed = 0
        total_failed = 0
        
        for module_dir in module_dirs:
            passed, failed = self.verify_module(module_dir.name, quick)
            total_passed += passed
            total_failed += failed
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Verification Summary:")
        print(f"   ✅ Passed: {total_passed}")
        print(f"   ❌ Failed: {total_failed}")
        
        if self.warnings and self.verbose:
            print(f"   ⚠️  Warnings: {len(self.warnings)}")
        
        if total_failed > 0:
            print("\n❌ Verification failed with errors")
            if not self.verbose:
                print("   Run with --verbose to see details")
            return 1
        
        if self.warnings and self.verbose:
            print(f"\n⚠️  {len(self.warnings)} warnings found (non-blocking)")
        
        print("\n✅ All examples verified successfully!")
        return 0


def main():
    """Main entry point for verification script."""
    parser = argparse.ArgumentParser(
        description="Verify Quantum Computing 101 example files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_examples.py                    # Verify all examples
  python verify_examples.py --module module1   # Verify specific module  
  python verify_examples.py --quick            # Quick syntax check only
  python verify_examples.py --verbose          # Show detailed output
        """
    )
    
    parser.add_argument(
        "--module",
        help="Verify specific module (e.g., module1_fundamentals)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: syntax check only, skip structure validation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including warnings"
    )
    
    args = parser.parse_args()
    
    verifier = ExampleVerifier(verbose=args.verbose)
    result = verifier.verify_all(module_filter=args.module, quick=args.quick)
    
    return result


if __name__ == "__main__":
    sys.exit(main())

