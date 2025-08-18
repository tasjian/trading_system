#!/bin/bash

# Trading System Android APK Build Script
# This script builds the Android APK for the trading system frontend

echo "🚀 Building Trading System Android APK..."

# Check if Android SDK is available
if [ -z "$ANDROID_HOME" ]; then
    echo "❌ ANDROID_HOME environment variable is not set"
    echo "Please install Android SDK and set ANDROID_HOME"
    echo "Example: export ANDROID_HOME=/path/to/android/sdk"
    exit 1
fi

# Check if Java is available
if ! command -v java &> /dev/null; then
    echo "❌ Java is not installed or not in PATH"
    echo "Please install Java 8 or higher"
    exit 1
fi

# Check if gradlew exists and is executable
if [ ! -x "./gradlew" ]; then
    echo "❌ gradlew is not executable"
    chmod +x ./gradlew
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
./gradlew clean

# Build debug APK
echo "🔨 Building debug APK..."
./gradlew assembleDebug

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo "✅ APK build successful!"
    echo "📱 APK location: app/build/outputs/apk/debug/app-debug.apk"
    
    # Check if APK exists
    if [ -f "app/build/outputs/apk/debug/app-debug.apk" ]; then
        APK_SIZE=$(du -h "app/build/outputs/apk/debug/app-debug.apk" | cut -f1)
        echo "📏 APK size: $APK_SIZE"
        echo ""
        echo "🎉 Ready to install on Android device!"
        echo "   1. Enable 'Unknown Sources' in Android Settings"
        echo "   2. Transfer APK to device"
        echo "   3. Open APK file to install"
    else
        echo "❌ APK file not found after build"
        exit 1
    fi
else
    echo "❌ APK build failed!"
    echo "Check the error messages above for details"
    exit 1
fi