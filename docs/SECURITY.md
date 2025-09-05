# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions of Quantum Computing 101:

| Version | Supported          |
| ------- | ------------------ |
| main    | ✅ Yes            |
| v1.x    | ✅ Yes            |
| < v1.0  | ❌ No             |

## Security Considerations for Quantum Computing Education

### Educational vs Production Context

**Important**: Quantum Computing 101 is designed as an educational platform. The examples and implementations are optimized for learning rather than production security. Please consider the following:

#### ✅ Safe for Educational Use
- All examples are designed for local simulation
- No sensitive data processing in standard examples
- Focus on quantum algorithm education
- Safe for classroom and personal learning environments

#### ⚠️ Production Considerations
- Examples are not hardened for production environments
- Cloud platform examples require your own API credentials
- Real quantum hardware access uses your own accounts
- Some examples demonstrate cryptographic concepts for educational purposes only

## Potential Security Areas

### 1. Cloud Platform Integration
**Scope**: Examples in Module 7 demonstrate IBM Quantum and AWS Braket integration

**Security Considerations**:
- API keys and tokens should never be committed to version control
- Use environment variables or secure credential storage
- Follow cloud provider security best practices
- Monitor usage to prevent unexpected charges

**Safe Usage**:
```bash
# Good: Use environment variables
export IBM_QUANTUM_TOKEN="your_token_here"
python module7_hardware/01_ibm_quantum_access.py

# Bad: Never hardcode credentials
# token = "your_actual_token"  # Don't do this!
```

### 2. File System Access
**Scope**: Examples may create temporary files and save visualizations

**Security Considerations**:
- Files are created in current directory by default
- Some examples may download small datasets
- Visualization files (PNG, PDF) are saved locally

**Safe Usage**:
- Run examples in dedicated directories
- Review file outputs periodically
- Use `--output-dir` parameter when available

### 3. Network Access
**Scope**: Cloud platform examples and potential package updates

**Security Considerations**:
- IBM Quantum and AWS Braket examples make authenticated API calls
- Package managers (pip) may download dependencies
- Quantum hardware access requires internet connectivity

**Safe Usage**:
- Use secure networks for quantum cloud access
- Keep dependencies updated
- Review network activity if running in sensitive environments

### 4. Quantum Cryptography Examples
**Scope**: Module 8 includes quantum cryptography demonstrations

**Security Considerations**:
- Examples are for educational purposes only
- Not suitable for actual cryptographic applications
- Demonstrate concepts rather than secure implementations

**Educational Context**:
- Quantum key distribution (QKD) examples show protocols
- Post-quantum cryptography examples demonstrate concepts
- Not intended for securing real communications

## Reporting Security Vulnerabilities

### How to Report

If you discover a security vulnerability, please report it responsibly:

#### 🔒 Private Disclosure (Preferred)
- **Email**: aicomputing101@gmail.com
- **Subject**: "Security Vulnerability Report"
- **Include**: Detailed description, steps to reproduce, potential impact

#### 📋 GitHub Security Advisory
- Use GitHub's [Security Advisory](https://github.com/AIComputing101/quantum-computing-101/security/advisories/new) feature for private reporting
- This allows coordinated disclosure and tracking

### What to Include

**Good Security Reports Include**:
- Clear description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested mitigation if known
- Your environment details (OS, Python version, etc.)

**Example Report**:
```
Subject: Security Vulnerability Report

Description: 
API credentials are logged in verbose mode in module7_hardware/01_ibm_quantum_access.py

Steps to Reproduce:
1. Set IBM_QUANTUM_TOKEN environment variable
2. Run: python 01_ibm_quantum_access.py --verbose
3. Observe token appears in console output

Potential Impact:
- Credential exposure in logs
- Possible unauthorized account access

Suggested Fix:
- Mask credentials in verbose output
- Use logging levels to control sensitive information
```

### What NOT to Report

The following are **not** security vulnerabilities for our educational context:

- ❌ **Quantum algorithm demonstrations**: Educational implementations aren't production-ready
- ❌ **Simulation limitations**: Classical simulators have known constraints
- ❌ **Academic algorithm details**: We implement textbook versions for learning
- ❌ **Performance issues**: Educational code prioritizes clarity over optimization
- ❌ **Missing production features**: Error handling may be simplified for learning

## Response Process

### Our Commitment
- **24-hour acknowledgment**: We'll confirm receipt within 24 hours
- **1-week initial assessment**: Initial impact analysis within one week
- **Coordinated disclosure**: We'll work with you on appropriate disclosure timing
- **Public credit**: Security researchers will be credited (unless they prefer anonymity)

### Timeline
1. **Report received**: Acknowledgment sent within 24 hours
2. **Investigation**: 1-7 days for initial assessment
3. **Fix development**: Timeline depends on severity and complexity
4. **Testing**: Verify fix doesn't break educational functionality
5. **Release**: Security fix released with appropriate severity level
6. **Disclosure**: Public disclosure after fix is available

## Security Best Practices for Users

### For Students and Educators

#### ✅ Recommended Practices
- Run examples in isolated directories
- Use virtual environments for Python dependencies
- Keep Qiskit and other packages updated
- Don't share API credentials or tokens
- Use version control for your modifications (exclude credentials)

#### 🔒 Credential Management
```bash
# Create .env file (add to .gitignore)
echo "IBM_QUANTUM_TOKEN=your_token" > .env
echo ".env" >> .gitignore

# Use in examples
source .env
python module7_hardware/01_ibm_quantum_access.py
```

### For Institutional Use

#### 🏫 Classroom Setup
- Use sandboxed environments for student access
- Provide shared credentials through secure channels
- Monitor cloud usage to prevent unexpected costs
- Regular security updates and reviews

#### 🔧 IT Administrator Guidelines
- Network segmentation for quantum cloud access
- Regular dependency scanning and updates
- Log monitoring for unusual activity
- Backup and recovery procedures

## Dependency Security

### Monitoring
We regularly monitor our dependencies for security issues using:
- GitHub Security Advisories
- Dependabot alerts
- Manual security reviews

### Updates
- Security updates are applied promptly
- Breaking changes are tested against all examples
- Release notes include security-relevant changes

### Key Dependencies
- **Qiskit**: Primary quantum computing framework
- **NumPy/SciPy**: Scientific computing (potential for numerical vulnerabilities)
- **Matplotlib**: Visualization (generally low risk)
- **Requests**: HTTP library for cloud access (monitor for vulnerabilities)

## Incident Response

### Classification

#### 🔴 Critical
- Credential exposure
- Arbitrary code execution
- Data exfiltration

#### 🟡 Medium
- Information disclosure
- Denial of service
- Privilege escalation

#### 🟢 Low
- Minor information leaks
- Performance issues
- Educational content accuracy

### Response Actions
1. **Immediate**: Disable affected functionality if critical
2. **Short-term**: Develop and test fix
3. **Long-term**: Improve security practices and testing

## Contact Information

### Security Team
- **Primary Contact**: aicomputing101@gmail.com
- **GitHub Security**: Use repository security advisories
- **Response Time**: 24-hour acknowledgment commitment

### General Support
- **Questions**: Use GitHub Discussions for non-security questions
- **Bug Reports**: Use GitHub Issues for non-security bugs
- **Educational Support**: aicomputing101@gmail.com

---

## Responsible Disclosure

We believe in responsible disclosure and appreciate security researchers who:
- Report vulnerabilities privately first
- Allow reasonable time for fixes
- Don't exploit vulnerabilities maliciously
- Help improve security for the entire quantum computing education community

**Thank you for helping keep Quantum Computing 101 secure for everyone!** 🔒🚀⚛️
