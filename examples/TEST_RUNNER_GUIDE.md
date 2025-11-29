# Quantum Computing 101 - Test Runner Guide

## Overview

The `test_all_examples.py` script automatically runs all example scripts in the course and reports which ones succeed and which fail. This helps you quickly identify any issues across the entire codebase without manually running each example.

## Quick Start

```bash
# Run all examples (default 60s timeout per script)
python3 test_all_examples.py

# Run all examples and continue even if some fail
python3 test_all_examples.py --continue

# Run only a specific module
python3 test_all_examples.py --module module1_fundamentals
```

## Usage Options

### Basic Options

- **No arguments**: Run all examples with default settings
  ```bash
  python3 test_all_examples.py
  ```

- **`--module MODULE`**: Run only examples from a specific module
  ```bash
  python3 test_all_examples.py --module module5_error_correction
  ```

- **`--timeout SECONDS`**: Set maximum execution time per script (default: 60)
  ```bash
  python3 test_all_examples.py --timeout 120
  ```

- **`--verbose, -v`**: Show detailed output from each script
  ```bash
  python3 test_all_examples.py --verbose
  ```

- **`--continue, -c`**: Continue testing even if a test fails
  ```bash
  python3 test_all_examples.py --continue
  ```

- **`--save-logs`**: Save logs for all tests (not just failures)
  ```bash
  python3 test_all_examples.py --save-logs
  ```

- **`--quick`**: Run quick tests only (30s timeout, useful for rapid iteration)
  ```bash
  python3 test_all_examples.py --quick
  ```

### Common Use Cases

1. **Test Everything Before Commit**
   ```bash
   python3 test_all_examples.py --continue
   ```
   This runs all tests and shows you all failures at once.

2. **Debug a Specific Module**
   ```bash
   python3 test_all_examples.py --module module3_programming --verbose
   ```
   This shows detailed output for debugging.

3. **Quick Sanity Check**
   ```bash
   python3 test_all_examples.py --quick --continue
   ```
   Fast check to see if anything is obviously broken.

4. **Test After Dependency Update**
   ```bash
   python3 test_all_examples.py --timeout 120 --continue --save-logs
   ```
   Comprehensive test with longer timeout and full logging.

## Output

### During Execution

The script shows:
- Progress bar with current test number
- Real-time pass/fail status with colored indicators
  - ✓ (green) = passed
  - ✗ (red) = failed
- Execution time for each test
- Brief error messages for failures

Example:
```
Testing module1_fundamentals (8 examples)
--------------------------------------------------------------------------------

[1/54] ✓ 01_classical_vs_quantum_bits.py (2.34s)
[2/54] ✓ 02_quantum_gates_circuits.py (1.87s)
[3/54] ✗ 03_superposition_measurement.py (0.45s)
  Error: ModuleNotFoundError: No module named 'qiskit'
  Full log: test_logs/module1_fundamentals_03_superposition_measurement.log
```

### Summary Report

After all tests complete, you'll see:
- Total tests run
- Number passed/failed
- Pass rate percentage
- Total execution time
- Fastest and slowest tests
- List of failed tests with log file locations

### Output Files

1. **`test_logs/`** directory: Contains detailed logs for failed tests
   - Format: `{module_name}_{script_name}.log`
   - Includes full stdout and stderr output
   - Automatically created only for failures (unless `--save-logs` is used)

2. **`test_results.json`**: Machine-readable test results
   - JSON format with all test results
   - Useful for CI/CD integration or further analysis
   - Includes timestamps, execution times, and pass/fail status

## Understanding Test Results

### Exit Codes

- `0`: All tests passed ✅
- `1`: One or more tests failed ❌

This makes it easy to use in CI/CD pipelines:
```bash
python3 test_all_examples.py && echo "All tests passed!" || echo "Tests failed!"
```

### Common Failure Reasons

1. **Missing Dependencies**
   ```
   Error: ModuleNotFoundError: No module named 'qiskit'
   ```
   **Solution**: Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Timeout Errors**
   ```
   Error: TIMEOUT: Script exceeded 60s time limit
   ```
   **Solution**: Increase timeout or use `--quick` to skip intensive tests
   ```bash
   python3 test_all_examples.py --timeout 120
   ```

3. **Hardware/API Errors**
   ```
   Error: IBMAccountError: Could not connect to IBM Quantum
   ```
   **Solution**: Some examples require API keys or specific hardware access. These can be skipped or run separately.

4. **Import Errors**
   ```
   Error: ImportError: cannot import name 'X' from 'qiskit'
   ```
   **Solution**: Version mismatch - update dependencies
   ```bash
   pip3 install --upgrade -r requirements.txt
   ```

## Tips and Best Practices

1. **Before Pushing Code**: Always run with `--continue` to see all failures
   ```bash
   python3 test_all_examples.py --continue
   ```

2. **When Debugging**: Use `--verbose` to see full output
   ```bash
   python3 test_all_examples.py --module module1_fundamentals --verbose
   ```

3. **For CI/CD**: Use longer timeout and continue on errors
   ```bash
   python3 test_all_examples.py --timeout 180 --continue --save-logs
   ```

4. **Quick Iteration**: Use `--quick` during development
   ```bash
   python3 test_all_examples.py --quick --module module1_fundamentals
   ```

5. **Check Logs**: When a test fails, check the log file for details
   ```bash
   cat test_logs/module1_fundamentals_01_classical_vs_quantum_bits.log
   ```

## Module Structure

The test runner automatically discovers examples in these modules:

- `module1_fundamentals/` - Basic quantum computing concepts
- `module2_mathematics/` - Mathematical foundations
- `module3_programming/` - Quantum programming
- `module4_algorithms/` - Quantum algorithms
- `module5_error_correction/` - Error correction and mitigation
- `module6_machine_learning/` - Quantum machine learning
- `module7_hardware/` - Hardware integration
- `module8_applications/` - Real-world applications

The `utils/` directory is automatically excluded from testing.

## Troubleshooting

### All Tests Timeout

If all tests timeout, you might need to:
1. Check if your system is under heavy load
2. Increase the timeout significantly: `--timeout 300`
3. Run modules separately to identify the problematic one

### Permission Denied

If you get "Permission denied":
```bash
chmod +x test_all_examples.py
```

### Out of Memory

Some quantum simulations require significant memory. If you run out:
1. Close other applications
2. Use `--quick` mode
3. Run modules one at a time
4. Skip intensive examples (like large VQE or deep circuits)

### Colors Not Showing

If you don't see colors in the output:
- Your terminal might not support ANSI colors
- Try a different terminal (most modern terminals support colors)
- The functionality still works; it's just less pretty!

## Integration with CI/CD

Example GitHub Actions workflow:

```yaml
name: Test Quantum Examples
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r examples/requirements.txt
      - name: Run tests
        run: python3 examples/test_all_examples.py --continue --timeout 180
      - name: Upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: test-logs
          path: examples/test_logs/
```

## Contributing

When adding new examples:
1. Place them in the appropriate `moduleX_*/` directory
2. Name them with a number prefix: `01_example_name.py`
3. Ensure they run without user interaction
4. Run the test suite before committing
5. Add appropriate error handling

The test runner will automatically discover and test new examples!

## Support

If you encounter issues with the test runner itself (not the examples):
1. Check this guide for solutions
2. Review the test logs in `test_logs/`
3. Check the JSON results in `test_results.json`
4. Run with `--verbose` for more details

Happy testing! 🚀

