# Testing Guide for Quantum Computing 101

This guide explains how to use the automated test runner to verify all examples work correctly.

## Quick Start

```bash
# From the project root, run all examples:
./test-examples.sh --continue

# Or from the examples directory:
cd examples
python3 test_all_examples.py --continue

# Test a specific module:
./test-examples.sh --module module1_fundamentals

# Quick test with shorter timeout:
./test-examples.sh --quick --continue
```

## What's Been Created

### 1. `examples/test_all_examples.py`
The main test runner script that:
- ✅ Automatically discovers all example scripts across all modules
- ✅ Runs each script with configurable timeout
- ✅ Captures stdout and stderr for each test
- ✅ Shows colored progress indicators (✓ for pass, ✗ for fail)
- ✅ Reports execution time for each script
- ✅ Generates detailed failure logs
- ✅ Saves machine-readable JSON results
- ✅ Calculates pass rates and statistics
- ✅ Identifies fastest and slowest tests

### 2. `test-examples.sh`
A convenient wrapper script to run tests from the project root directory.

### 3. `examples/TEST_RUNNER_GUIDE.md`
Comprehensive documentation with:
- All command-line options explained
- Common use cases and examples
- Troubleshooting guide
- CI/CD integration examples
- Tips and best practices

### 4. `examples/test_results.json`
Automatically generated JSON file containing:
- List of all passed tests with execution times
- List of all failed tests with log file paths
- Overall statistics (pass rate, total time, etc.)
- Timestamps for tracking test runs

### 5. `examples/test_logs/` (created on failures)
Directory containing detailed logs for any failed tests:
- Full stdout output
- Full stderr output
- Timestamps
- Named as `{module}_{script}.log`

## Common Usage Examples

### 1. Test Everything (Recommended Before Commits)
```bash
./test-examples.sh --continue --timeout 120
```
This runs all tests and shows you all failures at once, with generous timeout.

### 2. Test a Specific Module You're Working On
```bash
./test-examples.sh --module module3_programming --verbose
```
Shows detailed output for debugging.

### 3. Quick Sanity Check
```bash
./test-examples.sh --quick --continue
```
Runs all tests with shorter 30s timeout for rapid feedback.

### 4. Debug a Failing Test
```bash
# Run with verbose output
./test-examples.sh --module module5_error_correction --verbose

# Or check the log file
cat examples/test_logs/module5_error_correction_01_quantum_noise_models.log
```

## Test Results Example

After running tests, you'll see a summary like:

```
================================================================================
Test Summary
================================================================================

Total tests run:     54
Passed:              52
Failed:              2
Pass rate:           96.3%
Total execution time: 145.23s

Fastest test:        07_no_cloning_theorem.py (0.31s)
Slowest test:        05_first_quantum_algorithm.py (13.40s)

Failed Tests:
  ✗ module7_hardware/01_ibm_quantum_access.py
    Log: test_logs/module7_hardware_01_ibm_quantum_access.log
  ✗ module7_hardware/02_aws_braket_integration.py
    Log: test_logs/module7_hardware_02_aws_braket_integration.log
```

## Understanding Results

### Exit Codes
- `0` = All tests passed ✅
- `1` = One or more tests failed ❌

This makes the script perfect for CI/CD pipelines:
```bash
./test-examples.sh --continue && echo "Ready to deploy!" || echo "Fix tests first!"
```

### Color Coding
- 🟢 Green ✓ = Test passed
- 🔴 Red ✗ = Test failed
- 🔵 Blue = Module headers
- 🟡 Yellow = Warnings and error snippets
- 🟣 Purple = Section headers

### Performance Tracking
Each test shows execution time, and the summary includes:
- Total execution time across all tests
- Fastest test (good for finding lightweight examples)
- Slowest test (good for identifying intensive computations)

## Module Coverage

The test runner automatically tests all examples in:

1. ✅ **module1_fundamentals** (8 examples)
   - Classical vs quantum bits, gates, superposition, entanglement, etc.

2. ✅ **module2_mathematics** (5 examples)
   - Complex numbers, linear algebra, tensor products, etc.

3. ✅ **module3_programming** (6 examples)
   - Qiskit programming, framework comparison, circuit patterns, etc.

4. ✅ **module4_algorithms** (5 examples)
   - Deutsch-Jozsa, Grover's, QFT, Shor's, VQE

5. ✅ **module5_error_correction** (8 examples)
   - Noise models, Steane code, error mitigation, fault tolerance, etc.

6. ✅ **module6_machine_learning** (5 examples)
   - Feature maps, VQC, quantum neural networks, PCA, etc.

7. ✅ **module7_hardware** (5 examples)
   - IBM Quantum, AWS Braket, hardware optimization, etc.

8. ✅ **module8_applications** (6 examples)
   - Chemistry, finance, logistics, materials, cryptography, etc.

**Total: 54 example scripts** (utils excluded)

## Tips for Best Results

### 1. Install All Dependencies First
```bash
pip install -r examples/requirements.txt
```

### 2. Some Tests May Require API Keys
Tests that access real quantum hardware (IBM, AWS) may fail without credentials. This is expected. You can:
- Skip those modules
- Set up API keys (see individual example documentation)
- Accept that those tests will fail (they're marked clearly)

### 3. Use Appropriate Timeouts
- Default 60s: Good for most examples
- 90-120s: Better for algorithm-heavy modules
- 30s (--quick): Fast feedback during development
- 180s+: Comprehensive testing including slow examples

### 4. Monitor Resource Usage
Some quantum simulations are memory-intensive:
- Watch your RAM usage
- Close unnecessary applications
- Consider testing modules separately if you have limited RAM

### 5. Regular Testing
Run tests:
- ✅ Before committing changes
- ✅ After updating dependencies
- ✅ When debugging issues
- ✅ After adding new examples

## CI/CD Integration

The test runner is designed for easy CI/CD integration:

### GitHub Actions Example
```yaml
name: Test Examples
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r examples/requirements.txt
      
      - name: Run example tests
        run: |
          ./test-examples.sh --continue --timeout 120
      
      - name: Upload test logs
        if: failure()
        uses: actions/upload-artifact@v3
        with:
          name: test-logs
          path: examples/test_logs/
      
      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: examples/test_results.json
```

## Troubleshooting

### "Command not found" Error
Make sure the script is executable:
```bash
chmod +x test-examples.sh
chmod +x examples/test_all_examples.py
```

### All Tests Timeout
- Increase timeout: `./test-examples.sh --timeout 300`
- Check system load: `top` or `htop`
- Test modules individually to isolate issues

### Import Errors
Install/update dependencies:
```bash
pip install --upgrade -r examples/requirements.txt
```

### Memory Errors
- Close other applications
- Test modules separately
- Use `--quick` mode to skip intensive examples

### No Colors in Output
Your terminal might not support ANSI colors. The functionality works the same, just less colorful!

## Example Test Session

Here's what a successful test run looks like:

```bash
$ ./test-examples.sh --module module1_fundamentals

================================================================================
Quantum Computing 101 - Example Test Runner
================================================================================

Discovering examples...
Found 8 examples across 1 modules
Timeout per script: 60s


Testing module1_fundamentals (8 examples)
--------------------------------------------------------------------------------

[1/8] ✓ 01_classical_vs_quantum_bits.py (2.90s)
[2/8] ✓ 02_quantum_gates_circuits.py (3.16s)
[3/8] ✓ 03_superposition_measurement.py (1.93s)
[4/8] ✓ 04_quantum_entanglement.py (1.47s)
[5/8] ✓ 05_first_quantum_algorithm.py (13.40s)
[6/8] ✓ 06_quantum_teleportation.py (1.83s)
[7/8] ✓ 07_no_cloning_theorem.py (0.60s)
[8/8] ✓ 08_hardware_reality_check.py (0.77s)

================================================================================
Test Summary
================================================================================

Total tests run:     8
Passed:              8
Failed:              0
Pass rate:           100.0%
Total execution time: 26.06s

Fastest test:        07_no_cloning_theorem.py (0.60s)
Slowest test:        05_first_quantum_algorithm.py (13.40s)

Results saved to: /home/ysha/quantum-computing-101/examples/test_results.json

All tests passed! 🎉
```

## Advanced Features

### JSON Results for Analysis
Parse `test_results.json` for custom reporting:
```python
import json

with open('examples/test_results.json', 'r') as f:
    results = json.load(f)

print(f"Pass rate: {results['pass_rate']}%")
print(f"Slowest tests:")
for test in sorted(results['passed'], key=lambda x: x['time'], reverse=True)[:5]:
    print(f"  {test['script']}: {test['time']:.2f}s")
```

### Continuous Monitoring
Run tests periodically:
```bash
# Test every hour
watch -n 3600 './test-examples.sh --quick --continue'
```

### Integration with Make
Add to your `Makefile`:
```makefile
.PHONY: test
test:
	./test-examples.sh --continue

.PHONY: test-quick
test-quick:
	./test-examples.sh --quick --continue

.PHONY: test-module
test-module:
	./test-examples.sh --module $(MODULE) --verbose
```

Then use:
```bash
make test
make test-quick
make test-module MODULE=module1_fundamentals
```

## Contributing New Examples

When adding new examples to the course:

1. **Place in correct module directory**
   ```
   examples/moduleX_name/NN_example_name.py
   ```

2. **Test it works standalone**
   ```bash
   python3 examples/moduleX_name/NN_example_name.py
   ```

3. **Run the test suite**
   ```bash
   ./test-examples.sh --module moduleX_name
   ```

4. **Run full suite before committing**
   ```bash
   ./test-examples.sh --continue
   ```

The test runner automatically discovers new examples - no configuration needed!

## Summary

The test runner provides:
- 🚀 **Automation** - Test all 54 examples with one command
- 📊 **Reporting** - Clear pass/fail status with statistics
- 🔍 **Debugging** - Detailed logs for failures
- ⚡ **Speed** - Quick mode for rapid iteration
- 🎯 **Focus** - Test specific modules during development
- 📈 **Tracking** - JSON results for analysis and trends
- 🌈 **UX** - Color-coded output for easy scanning
- 🔧 **Flexibility** - Many options for different use cases

Happy testing! 🎉

