#!/bin/bash
# Fix nimpy OpenGL environment to match native Nim execution

echo "🔧 Fixing nimpy OpenGL environment..."

# Detect platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Applying macOS-specific OpenGL fixes..."
    
    # macOS OpenGL framework paths
    export DYLD_FRAMEWORK_PATH="/System/Library/Frameworks:${DYLD_FRAMEWORK_PATH}"
    export DYLD_LIBRARY_PATH="/usr/lib:${DYLD_LIBRARY_PATH}"
    
    # macOS Metal/OpenGL compatibility
    export DYLD_FALLBACK_LIBRARY_PATH="/usr/lib:/System/Library/Frameworks/OpenGL.framework:${DYLD_FALLBACK_LIBRARY_PATH}"
    
    echo "✅ macOS OpenGL paths configured"
    
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Applying Linux-specific OpenGL fixes..."
    
    # Linux OpenGL library paths
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:/opt/homebrew/lib:${LD_LIBRARY_PATH}"
    
    # X11 display (if needed)
    if [ -z "$DISPLAY" ]; then
        export DISPLAY=":0"
    fi
    
    echo "✅ Linux OpenGL paths configured"
fi

# Common environment fixes
echo "🔗 Applying common environment fixes..."

# Ensure tribal directory is in Python path
TRIBAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/tribal" && pwd)"
export PYTHONPATH="${TRIBAL_DIR}:${PYTHONPATH}"

# Set working directory to tribal (important for asset loading)
cd "${TRIBAL_DIR}"

echo "📁 Working directory: $(pwd)"
echo "🐍 PYTHONPATH includes: ${TRIBAL_DIR}"

# Test the fixed environment
echo "🧪 Testing fixed environment..."

# First test native Nim to establish baseline
echo "1️⃣  Testing native Nim execution..."
if timeout 5s nim c -r -d:release src/tribal 2>/dev/null; then
    echo "✅ Native Nim OpenGL works with current environment"
else
    echo "⚠️  Native Nim may have issues, but continuing..."
fi

# Test Python with fixed environment
echo "2️⃣  Testing Python nimpy with fixed environment..."
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    import tribal_nimpy_viewer as viewer
    print('✅ Nimpy import successful')
    
    # Test initialization
    if viewer.initVisualization():
        print('✅ OpenGL initialization successful!')
        viewer.closeVisualization()
    else:
        print('❌ OpenGL initialization failed')
except Exception as e:
    print(f'❌ Nimpy test failed: {e}')
"

echo ""
echo "🎯 Environment is now configured for nimpy OpenGL!"
echo "💡 Run your tribal commands from this terminal session."
echo ""
echo "🚀 To test the full system:"
echo "   uv run ./tools/run.py experiments.recipes.tribal_basic.play --args policy_uri=test_move"
echo ""
echo "🔍 To debug further:"
echo "   python debug_opengl_environment.py"