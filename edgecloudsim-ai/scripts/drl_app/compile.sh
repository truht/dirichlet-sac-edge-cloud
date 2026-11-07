#!/bin/sh
# compile.sh — Build EdgeCloudSim SAC/DRL offloading simulation
# Compiles DRLMainApp and all dependencies in drl + drl_app packages

rm -rf ../../bin
mkdir ../../bin

javac \
  -classpath "../../lib/cloudsim-7.0.0-alpha.jar:../../lib/commons-math3-3.6.1.jar:../../lib/colt.jar" \
  -sourcepath ../../src \
  ../../src/edu/boun/edgecloudsim/applications/drl_app/DRLMainApp.java \
  -d ../../bin

if [ $? -eq 0 ]; then
    echo "Compilation successful."
else
    echo "Compilation FAILED." >&2
    exit 1
fi
