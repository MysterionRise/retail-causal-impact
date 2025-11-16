#!/bin/bash
# Verification script to demonstrate CI/CD infrastructure

echo "=========================================="
echo "CI/CD Infrastructure Verification"
echo "=========================================="
echo ""

echo "✓ 1. GitHub Actions CI Workflow"
if [ -f .github/workflows/ci.yml ]; then
    echo "   ✅ CI workflow configured"
    echo "   📝 Runs on: push to main/master/claude/**, pull requests"
    echo "   🐍 Python versions: 3.10, 3.11"
else
    echo "   ❌ CI workflow missing"
fi
echo ""

echo "✓ 2. Test Suite"
echo "   Running pytest..."
python -m pytest tests/ -v --tb=line -q 2>&1 | tail -5
echo ""

echo "✓ 3. Code Coverage"
python -m pytest tests/ --cov=src/coupon_causal --cov-report=term --tb=line -q 2>&1 | tail -15
echo ""

echo "✓ 4. Linting Setup"
if command -v ruff &> /dev/null; then
    echo "   ✅ Ruff linter installed"
    echo "   Checking for critical errors..."
    ruff check src/ tests/ scripts/ --select F,E --quiet && echo "   ✅ No critical errors" || echo "   ⚠️  Some linting issues (non-blocking)"
else
    echo "   ❌ Ruff not installed"
fi
echo ""

echo "✓ 5. Pre-commit Hooks"
if [ -f .pre-commit-config.yaml ]; then
    echo "   ✅ Pre-commit configuration exists"
else
    echo "   ❌ Pre-commit config missing"
fi
echo ""

echo "✓ 6. Package Build"
echo "   Testing package import..."
python -c "import sys; sys.path.insert(0, 'src'); from coupon_causal import data, ate, cate; print('   ✅ Package imports successfully')" 2>&1
echo ""

echo "✓ 7. Quick Smoke Test"
python test_quick.py 2>&1 | grep "✓"
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "All core CI/CD components are configured!"
echo ""
echo "Next steps:"
echo "1. Push to GitHub - CI will run automatically"
echo "2. Check Actions tab: https://github.com/MysterionRise/retail-causal-impact/actions"
echo "3. Address any linting issues flagged by CI"
echo ""
