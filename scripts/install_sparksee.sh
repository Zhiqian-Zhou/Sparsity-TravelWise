#!/usr/bin/env bash
# scripts/install_sparksee.sh
# ----------------------------------------------------------------------------
# One-time, no-root install of Sparksee 5.2.3 + a portable OpenJDK 17 into
# .tools/ at the repo root. Both come from public, redistributable sources:
#
#   * Sparksee Java jar — Maven Central (com.sparsity:sparkseejava:5.2.3).
#     The jar already bundles the linux64 native libraries
#     (libsparksee.so, libsparkseejavawrap.so, libstlport.so).
#   * Eclipse Temurin 17 JDK — installed via the `install-jdk` PyPI helper.
#
# After this script finishes, `python stop_level/build_kg.py` (Sparksee is the
# only backend now) will pick the binaries up via env vars set in the same
# shell, so source-it-once and you're done:
#
#   source scripts/install_sparksee.sh
#   python stop_level/build_kg.py --rebuild
#
# Re-running the script is a no-op if both binaries already exist.
# ----------------------------------------------------------------------------
set -eu

# Resolve the script directory in both bash (when executed) and zsh (when
# `source`d, since `source` doesn't honour the shebang). `${(%):-%x}` is the
# zsh equivalent of `${BASH_SOURCE[0]}`. Fall back to $0 for plain `sh`.
if [ -n "${BASH_SOURCE:-}" ]; then
    _self="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
    _self="${(%):-%x}"
else
    _self="$0"
fi
ROOT="$(cd "$(dirname "$_self")/.." && pwd)"
TOOLS="$ROOT/.tools"
SPARKSEE_HOME="$TOOLS/sparksee"
JAR="$SPARKSEE_HOME/sparkseejava-5.2.3.jar"
JAR_URL="https://repo1.maven.org/maven2/com/sparsity/sparkseejava/5.2.3/sparkseejava-5.2.3.jar"

mkdir -p "$SPARKSEE_HOME"

# 1) Sparksee jar + native libs
if [[ -f "$JAR" && -d "$SPARKSEE_HOME/native" ]]; then
    echo "[sparksee] already installed at $SPARKSEE_HOME"
else
    echo "[sparksee] downloading $JAR_URL ..."
    curl --fail --silent --show-error -L -o "$JAR" "$JAR_URL"
    echo "[sparksee] extracting Linux native libs from jar ..."
    rm -rf "$SPARKSEE_HOME/native"
    mkdir -p "$SPARKSEE_HOME/native"
    unzip -j -o -q "$JAR" 'linux64.nativelibs/*.so' -d "$SPARKSEE_HOME/native"
    echo "[sparksee] installed at $SPARKSEE_HOME"
fi

# 2) Portable OpenJDK 17 (only if no JAVA_HOME / no `java` on PATH)
if [[ -n "${JAVA_HOME:-}" && -x "${JAVA_HOME}/bin/java" ]]; then
    echo "[jdk] using existing JAVA_HOME=$JAVA_HOME"
elif command -v java >/dev/null 2>&1; then
    echo "[jdk] using system java: $(command -v java)"
else
    EXISTING_JDK="$(find "$TOOLS" -maxdepth 1 -type d -name 'jdk-17*' 2>/dev/null | head -1 || true)"
    if [[ -n "$EXISTING_JDK" ]]; then
        export JAVA_HOME="$EXISTING_JDK"
        echo "[jdk] reusing $JAVA_HOME"
    else
        echo "[jdk] downloading Eclipse Temurin 17 (portable, no root) ..."
        ( cd "$TOOLS" && python3 -c "import jdk; print(jdk.install('17', path='.'))" )
        EXISTING_JDK="$(find "$TOOLS" -maxdepth 1 -type d -name 'jdk-17*' | head -1)"
        export JAVA_HOME="$EXISTING_JDK"
        echo "[jdk] installed at $JAVA_HOME"
    fi
fi

# 3) Export the env so `python stop_level/build_kg.py` picks them up
export SPARKSEE_HOME
export LD_LIBRARY_PATH="$SPARKSEE_HOME/native${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo
echo "✓ Sparksee 5.2.3 ready."
echo "  SPARKSEE_HOME = $SPARKSEE_HOME"
echo "  JAVA_HOME     = ${JAVA_HOME:-<system>}"
echo
echo "Now run:"
echo "  python stop_level/build_kg.py --rebuild"
