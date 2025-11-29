#!/usr/bin/env python3
"""
Test Runner for Quantum Computing 101 Examples

This script automatically runs all example scripts in the quantum computing course
and reports which ones succeed and which ones fail. It helps quickly identify
any issues across the entire codebase.

Usage:
    python3 test_all_examples.py                    # Run all examples
    python3 test_all_examples.py --module module1   # Run specific module
    python3 test_all_examples.py --timeout 120      # Set custom timeout
    python3 test_all_examples.py --verbose          # Show detailed output
    python3 test_all_examples.py --continue         # Continue on errors

Author: Quantum Computing 101 Course
License: MIT
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# ANSI color codes for terminal output
class Colors:
    """Terminal color codes for pretty output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def discover_examples(base_path: Path, module_filter: str = None) -> Dict[str, List[Path]]:
    """
    Discover all example Python files organized by module.
    
    Args:
        base_path: Root path to examples directory
        module_filter: Optional module name to filter (e.g., "module1")
    
    Returns:
        Dictionary mapping module names to lists of example file paths
    """
    examples = {}
    
    # Find all module directories
    for module_dir in sorted(base_path.glob("module*")):
        if not module_dir.is_dir():
            continue
        
        module_name = module_dir.name
        
        # Apply filter if specified
        if module_filter and module_name != module_filter:
            continue
        
        # Find all Python files in this module (excluding __pycache__)
        py_files = sorted([
            f for f in module_dir.glob("*.py")
            if not f.name.startswith("_")
        ])
        
        if py_files:
            examples[module_name] = py_files
    
    return examples


def run_example(
    script_path: Path,
    timeout: int = 60,
    verbose: bool = False
) -> Tuple[bool, float, str, str]:
    """
    Run a single example script and capture results.
    
    Args:
        script_path: Path to the Python script
        timeout: Maximum execution time in seconds
        verbose: Whether to print output in real-time
    
    Returns:
        Tuple of (success, execution_time, stdout, stderr)
    """
    start_time = time.time()
    
    try:
        # Run the script with timeout
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        execution_time = time.time() - start_time
        success = result.returncode == 0
        
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        
        return success, execution_time, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        error_msg = f"TIMEOUT: Script exceeded {timeout}s time limit"
        return False, execution_time, "", error_msg
    
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"EXCEPTION: {type(e).__name__}: {str(e)}"
        return False, execution_time, "", error_msg


def save_failure_log(script_path: Path, stdout: str, stderr: str, log_dir: Path):
    """Save detailed logs for failed tests."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename based on script path
    log_name = f"{script_path.parent.name}_{script_path.stem}.log"
    log_file = log_dir / log_name
    
    with open(log_file, 'w') as f:
        f.write(f"Failed Test Log: {script_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STDOUT:\n")
        f.write("-" * 80 + "\n")
        f.write(stdout if stdout else "(no output)\n")
        f.write("\n")
        
        f.write("STDERR:\n")
        f.write("-" * 80 + "\n")
        f.write(stderr if stderr else "(no errors)\n")
    
    return log_file


def print_progress_bar(current: int, total: int, prefix: str = '', suffix: str = '', length: int = 50):
    """Print a progress bar to the terminal."""
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(length * current // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if current == total:
        print()


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for all Quantum Computing 101 examples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Run all examples
  %(prog)s --module module1         Run only module1 examples
  %(prog)s --timeout 120            Set 120s timeout per script
  %(prog)s --verbose                Show detailed output
  %(prog)s --continue               Continue testing even if tests fail
  %(prog)s --save-logs              Save logs for all tests (not just failures)
        """
    )
    
    parser.add_argument(
        '--module',
        type=str,
        help='Run only examples from specified module (e.g., module1_fundamentals)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Maximum execution time per script in seconds (default: 60)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed output from each script'
    )
    parser.add_argument(
        '--continue', '-c',
        action='store_true',
        dest='continue_on_error',
        help='Continue testing even if a test fails'
    )
    parser.add_argument(
        '--save-logs',
        action='store_true',
        help='Save logs for all tests, not just failures'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (shorter timeout, skip intensive examples)'
    )
    
    args = parser.parse_args()
    
    # Adjust timeout for quick mode
    if args.quick:
        args.timeout = 30
    
    # Setup paths
    script_dir = Path(__file__).parent
    log_dir = script_dir / "test_logs"
    
    # Print header
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Quantum Computing 101 - Example Test Runner{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    # Discover examples
    print(f"{Colors.OKCYAN}Discovering examples...{Colors.ENDC}")
    examples = discover_examples(script_dir, args.module)
    
    if not examples:
        print(f"{Colors.FAIL}No examples found!{Colors.ENDC}")
        if args.module:
            print(f"Module filter: {args.module}")
        return 1
    
    # Count total examples
    total_examples = sum(len(files) for files in examples.values())
    
    print(f"Found {Colors.BOLD}{total_examples}{Colors.ENDC} examples across {Colors.BOLD}{len(examples)}{Colors.ENDC} modules")
    print(f"Timeout per script: {Colors.BOLD}{args.timeout}s{Colors.ENDC}\n")
    
    # Track results
    results = {
        'passed': [],
        'failed': [],
        'total_time': 0,
        'start_time': datetime.now()
    }
    
    current_test = 0
    
    # Run tests for each module
    for module_name, files in examples.items():
        print(f"\n{Colors.BOLD}{Colors.OKBLUE}Testing {module_name} ({len(files)} examples){Colors.ENDC}")
        print(f"{Colors.OKBLUE}{'-'*80}{Colors.ENDC}\n")
        
        for script_path in files:
            current_test += 1
            script_name = script_path.name
            
            # Print current test
            status_prefix = f"[{current_test}/{total_examples}]"
            print(f"{status_prefix} Testing {script_name}...", end='', flush=True)
            
            # Run the test
            success, exec_time, stdout, stderr = run_example(
                script_path,
                timeout=args.timeout,
                verbose=args.verbose
            )
            
            results['total_time'] += exec_time
            
            # Print result
            if success:
                print(f"\r{status_prefix} {Colors.OKGREEN}✓{Colors.ENDC} {script_name} ({exec_time:.2f}s)")
                results['passed'].append({
                    'module': module_name,
                    'script': script_name,
                    'time': exec_time
                })
                
                if args.save_logs:
                    save_failure_log(script_path, stdout, stderr, log_dir)
            else:
                print(f"\r{status_prefix} {Colors.FAIL}✗{Colors.ENDC} {script_name} ({exec_time:.2f}s)")
                
                # Save failure log
                log_file = save_failure_log(script_path, stdout, stderr, log_dir)
                
                results['failed'].append({
                    'module': module_name,
                    'script': script_name,
                    'time': exec_time,
                    'log': str(log_file)
                })
                
                # Show error snippet
                if not args.verbose and stderr:
                    error_lines = stderr.strip().split('\n')
                    print(f"  {Colors.WARNING}Error: {error_lines[-1][:100]}{Colors.ENDC}")
                    print(f"  {Colors.WARNING}Full log: {log_file}{Colors.ENDC}")
                
                # Stop if not continuing on error
                if not args.continue_on_error:
                    print(f"\n{Colors.FAIL}Stopping due to test failure. Use --continue to continue testing.{Colors.ENDC}")
                    break
        
        # Break outer loop if stopping on error
        if results['failed'] and not args.continue_on_error:
            break
    
    # Print summary
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}Test Summary{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*80}{Colors.ENDC}\n")
    
    total_run = len(results['passed']) + len(results['failed'])
    pass_rate = (len(results['passed']) / total_run * 100) if total_run > 0 else 0
    
    print(f"Total tests run:     {Colors.BOLD}{total_run}{Colors.ENDC}")
    print(f"Passed:              {Colors.OKGREEN}{len(results['passed'])}{Colors.ENDC}")
    print(f"Failed:              {Colors.FAIL if results['failed'] else Colors.OKGREEN}{len(results['failed'])}{Colors.ENDC}")
    print(f"Pass rate:           {Colors.OKGREEN if pass_rate == 100 else Colors.WARNING}{pass_rate:.1f}%{Colors.ENDC}")
    print(f"Total execution time: {Colors.BOLD}{results['total_time']:.2f}s{Colors.ENDC}")
    
    # Show fastest and slowest tests
    if results['passed']:
        fastest = min(results['passed'], key=lambda x: x['time'])
        slowest = max(results['passed'], key=lambda x: x['time'])
        print(f"\nFastest test:        {fastest['script']} ({fastest['time']:.2f}s)")
        print(f"Slowest test:        {slowest['script']} ({slowest['time']:.2f}s)")
    
    # List failed tests
    if results['failed']:
        print(f"\n{Colors.FAIL}{Colors.BOLD}Failed Tests:{Colors.ENDC}")
        for failure in results['failed']:
            print(f"  {Colors.FAIL}✗{Colors.ENDC} {failure['module']}/{failure['script']}")
            print(f"    Log: {failure['log']}")
    
    # Save results to JSON
    results_file = script_dir / "test_results.json"
    with open(results_file, 'w') as f:
        # Convert datetime to string for JSON serialization
        results_json = {
            'passed': results['passed'],
            'failed': results['failed'],
            'total_time': results['total_time'],
            'start_time': results['start_time'].isoformat(),
            'pass_rate': pass_rate,
            'total_tests': total_run
        }
        json.dump(results_json, f, indent=2)
    
    print(f"\n{Colors.OKCYAN}Results saved to: {results_file}{Colors.ENDC}")
    
    if results['failed']:
        print(f"\n{Colors.FAIL}Some tests failed. Review the logs for details.{Colors.ENDC}")
        return 1
    else:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}All tests passed! 🎉{Colors.ENDC}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

