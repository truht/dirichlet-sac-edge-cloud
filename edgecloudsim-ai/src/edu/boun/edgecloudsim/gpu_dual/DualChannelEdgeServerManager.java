/*
 * Title:        EdgeCloudSim - DualChannelEdgeServerManager  (Phase 2)
 *
 * Description:
 * Creates DualChannelEdgeVm instances (with DualChannelScheduler) instead
 * of plain EdgeVM.  Reads <cpu_mips>, <gpu_mips>, <vram_mb>, <power_w> from XML.
 *
 * XML additions per VM (all optional — sensible defaults provided):
 *   <cpu_mips>3000</cpu_mips>       ARM Cortex-A57 MIPS  (default: 3000)
 *   <gpu_mips>135000</gpu_mips>     TX2 GPU MFLOPS        (default: same as <mips>)
 *   <vram_mb>8192</vram_mb>         Unified LPDDR4 MB     (default: 8192)
 *   <power_w>15</power_w>           Device power budget W (default: 15) --
 *                                   only used by the DRL app's battery model
 *                                   (edu.boun.edgecloudsim.drl.EdgeEnergyManager),
 *                                   ignored by tutorial3_ai_flop's own orchestrator.
 *
 * Note: VM.mips (<mips>) must stay at GPU MFLOPS (135,000) so:
 *   1. getAvgUtilization() reflects GPU load (not tiny CPU load)
 *   2. CloudSim formula processingTime = effectiveMI / vmMips = effectiveTime ✓
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import java.util.ArrayList;
import java.util.List;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;
import edu.boun.edgecloudsim.edge_server.EdgeVM;

/**
 * Phase 2 edge server manager — creates {@link DualChannelEdgeVm} instances.
 *
 * <p>Inherits all datacenter/host creation and {@code getAvgUtilization()}
 * from {@link DefaultEdgeServerManager}. Only {@code createVmList()} is
 * overridden to inject the dual-channel scheduler and GPU fields.</p>
 */
public class DualChannelEdgeServerManager extends DefaultEdgeServerManager {

    /** ARM Cortex-A57 default CPU MIPS when <cpu_mips> absent from XML. */
    private static final long DEFAULT_CPU_MIPS = 3_000L;

    /** Default VRAM when <vram_mb> absent from XML (8 GB TX2 unified memory). */
    private static final long DEFAULT_VRAM_MB  = 8_192L;

    /** Default power budget when <power_w> absent from XML (TX2 ~15W). */
    private static final double DEFAULT_POWER_W = 15.0;

    @Override
    public void createVmList(int brokerId) {
        int hostCounter = 0;
        int vmCounter   = 0;

        Document doc = SimSettings.getInstance().getEdgeDevicesDocument();
        NodeList datacenterList = doc.getElementsByTagName("datacenter");

        for (int i = 0; i < datacenterList.getLength(); i++) {
            Element dcEl = (Element) datacenterList.item(i);
            NodeList hostNodeList = dcEl.getElementsByTagName("host");

            for (int j = 0; j < hostNodeList.getLength(); j++) {
                vmList.add(hostCounter, new ArrayList<>());

                Element hostEl = (Element) hostNodeList.item(j);
                NodeList vmNodeList = hostEl.getElementsByTagName("VM");

                for (int k = 0; k < vmNodeList.getLength(); k++) {
                    Element vmEl = (Element) vmNodeList.item(k);

                    String vmm     = vmEl.getAttribute("vmm");
                    int    cores   = Integer.parseInt(
                            vmEl.getElementsByTagName("core").item(0).getTextContent());
                    double mips    = Double.parseDouble(
                            vmEl.getElementsByTagName("mips").item(0).getTextContent());
                    int    ram     = Integer.parseInt(
                            vmEl.getElementsByTagName("ram").item(0).getTextContent());
                    long   storage = Long.parseLong(
                            vmEl.getElementsByTagName("storage").item(0).getTextContent());
                    long   bw      = SimSettings.getInstance().getWlanBandwidth()
                                     / (hostNodeList.getLength() + vmNodeList.getLength());

                    // Phase 2 GPU extensions (optional XML elements)
                    long cpuMips = parseLongTag(vmEl, "cpu_mips", DEFAULT_CPU_MIPS);
                    long gpuMips = parseLongTag(vmEl, "gpu_mips", (long) mips);
                    long vramMB  = parseLongTag(vmEl, "vram_mb",  DEFAULT_VRAM_MB);
                    double powerW = parseDoubleTag(vmEl, "power_w", DEFAULT_POWER_W);

                    // Use TimeShared to avoid SpaceShared queue build-up at high device counts.
                    // At >200 devices, edge ρ > 1 with SpaceShared → O(n²) slowdown.
                    // TimeShared runs all tasks concurrently sharing MIPS (models GPU batching).
                    DualChannelSchedulerTimeShared sched = new DualChannelSchedulerTimeShared(cpuMips);

                    DualChannelEdgeVm vm = new DualChannelEdgeVm(
                            vmCounter, brokerId, mips, cores, ram, bw, storage, vmm,
                            sched, cpuMips, gpuMips, vramMB, powerW);

                    vmList.get(hostCounter).add(vm);
                    vmCounter++;
                }
                hostCounter++;
            }
        }
    }

    /**
     * Override utilization reporting to use load-factor metric (runningTasks / totalPEs).
     *
     * Problem: CloudletSchedulerTimeShared.getTotalUtilizationOfCpu() reports ~18.77%
     * at 200 devices due to MIPS-sharing arithmetic, far below the 75% threshold used
     * by UTILIZATION_BASED in SampleEdgeOrchestrator. The algorithm never routes to cloud.
     *
     * Fix: report utilization as (concurrent tasks in execution / total PE slots) × 100,
     * which is the standard load factor ρ — identical in meaning to SpaceShared utilization
     * and consistent with the original 75% threshold design intent.
     *
     * At 200 devices (all at edge): ~33 concurrent tasks / 56 PEs = 59% → below threshold.
     * At ~260 devices: ~42 tasks / 56 PEs ≈ 75% → UTILIZATION_BASED starts routing to cloud.
     * UTILIZATION_BASED then self-regulates: cloud offload → edge load drops → stays near 75%.
     *
     * @return utilization percentage [0, 100] based on load factor
     */
    @Override
    public double getAvgUtilization() {
        int totalRunning = 0;
        int totalPEs = 0;
        for (List<EdgeVM> vms : vmList) {
            for (EdgeVM vm : vms) {
                totalRunning += vm.getCloudletScheduler().runningCloudlets();
                totalPEs += vm.getNumberOfPes();
            }
        }
        if (totalPEs == 0) {
			return 0.0;
		}
        return Math.min(100.0, totalRunning * 100.0 / totalPEs);
    }

    private static long parseLongTag(Element el, String tag, long def) {
        NodeList nl = el.getElementsByTagName(tag);
        if (nl.getLength() > 0 && nl.item(0) != null) {
            try { return Long.parseLong(nl.item(0).getTextContent().trim()); }
            catch (NumberFormatException ignored) {}
        }
        return def;
    }

    private static double parseDoubleTag(Element el, String tag, double def) {
        NodeList nl = el.getElementsByTagName(tag);
        if (nl.getLength() > 0 && nl.item(0) != null) {
            try { return Double.parseDouble(nl.item(0).getTextContent().trim()); }
            catch (NumberFormatException ignored) {}
        }
        return def;
    }
}
