#!/bin/sh
# runner.sh — Run the SAC/DRL offloading simulation.
#
# Unlike the other apps' runner.sh, there is no iteration_number argument:
# DRLMainApp itself loops over Max_Episodes internally (Algorithm 1), all
# within a single JVM process sharing one persistent socket connection to
# the Python drl_controller. Start drl_controller/train.py BEFORE running
# this script — DRLMainApp blocks waiting for it to connect (see
# /PROTOCOL.md section 1).

script_root_path="$(cd "$(dirname "$0")" && pwd)"

config_file="${script_root_path}/config/default_config.properties"
edge_devices_file="${script_root_path}/config/edge_devices.xml"
applications_file="${script_root_path}/config/applications.xml"
output_folder="${script_root_path}/../../sim_results/drl_app"
console_log="${output_folder}/console.log"

mkdir -p "$output_folder"
cd "$script_root_path"

java \
  -classpath "../../bin:../../lib/cloudsim-7.0.0-alpha.jar:../../lib/commons-math3-3.6.1.jar:../../lib/colt.jar" \
  edu.boun.edgecloudsim.applications.drl_app.DRLMainApp \
  "$config_file" \
  "$edge_devices_file" \
  "$applications_file" \
  "$output_folder" \
  2>&1 | tee -a "$console_log"
