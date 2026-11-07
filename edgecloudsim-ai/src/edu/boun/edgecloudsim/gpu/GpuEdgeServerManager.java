/*
 * Title:        EdgeCloudSim - GpuEdgeServerManager
 *
 * Description:
 * Extends DefaultEdgeServerManager to create GpuVm instances
 * (instead of plain EdgeVM) for each VM defined in edge_devices.xml.
 *
 * Reads two extra optional XML elements per VM:
 *   <gpu_mips>135000</gpu_mips>   — GPU MFLOPS (default: same as <mips>)
 *   <vram_mb>8192</vram_mb>       — Unified memory in MB (default: 8192)
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu;

import java.util.ArrayList;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.edge_server.DefaultEdgeServerManager;

/**
 * GPU-aware edge server manager.
 *
 * <p>Overrides {@link DefaultEdgeServerManager#createVmList(int)} to produce
 * {@link GpuVm} instances (each paired with a {@link GpuCloudletScheduler})
 * instead of plain {@code EdgeVM}.</p>
 *
 * <p><b>XML format (additions to standard edge_devices.xml):</b>
 * <pre>{@code
 * <VM vmm="Xen">
 *   <core>4</core>
 *   <mips>135000</mips>       <!-- kept: serves as vmMips reference in GpuCloudletScheduler -->
 *   <ram>4096</ram>
 *   <storage>32000</storage>
 *   <gpu_mips>135000</gpu_mips>   <!-- GPU MFLOPS — copied to GpuVm + GpuCloudletScheduler -->
 *   <vram_mb>8192</vram_mb>        <!-- VRAM in MB for admission control -->
 * </VM>
 * }</pre>
 * If {@code <gpu_mips>} is absent, falls back to the value of {@code <mips>}.
 * If {@code <vram_mb>} is absent, defaults to 8192 MB.</p>
 *
 * <p><b>Compatibility:</b> All other methods ({@code startDatacenters()},
 * {@code getAvgUtilization()}, {@code terminateDatacenters()}) are inherited
 * unchanged from {@code DefaultEdgeServerManager}.</p>
 */
public class GpuEdgeServerManager extends DefaultEdgeServerManager {

    /**
     * Default VRAM in MB when not specified in XML.
     * 8,192 MB = 8 GB unified LPDDR4 on Jetson TX2.
     */
    private static final long DEFAULT_VRAM_MB = 8_192L;

    /**
     * Creates VM lists exactly as the parent does, but produces {@link GpuVm}
     * instances paired with {@link GpuCloudletScheduler}.
     *
     * @param brokerId ID of the broker managing the created VMs
     */
    @Override
    public void createVmList(int brokerId) {
        int hostCounter = 0;
        int vmCounter   = 0;

        Document doc = SimSettings.getInstance().getEdgeDevicesDocument();
        NodeList datacenterList = doc.getElementsByTagName("datacenter");

        for (int i = 0; i < datacenterList.getLength(); i++) {
            Node datacenterNode = datacenterList.item(i);
            Element dcElement   = (Element) datacenterNode;
            NodeList hostNodeList = dcElement.getElementsByTagName("host");

            for (int j = 0; j < hostNodeList.getLength(); j++) {
                // Initialise VM list for this host
                vmList.add(hostCounter, new ArrayList<>());

                Node hostNode   = hostNodeList.item(j);
                Element hostEl  = (Element) hostNode;
                NodeList vmNodeList = hostEl.getElementsByTagName("VM");

                for (int k = 0; k < vmNodeList.getLength(); k++) {
                    Node vmNode   = vmNodeList.item(k);
                    Element vmEl  = (Element) vmNode;

                    // Standard VM parameters
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

                    // GPU extensions — optional, with defaults
                    long gpuMips  = parseLongTag(vmEl, "gpu_mips",  (long) mips);
                    long vramMB   = parseLongTag(vmEl, "vram_mb",   DEFAULT_VRAM_MB);

                    // Create GPU-capable VM with dual-channel scheduler
                    GpuVm vm = new GpuVm(
                        vmCounter, brokerId, mips, cores, ram, bw, storage, vmm,
                        new GpuCloudletScheduler(gpuMips),
                        gpuMips, vramMB
                    );
                    vmList.get(hostCounter).add(vm);
                    vmCounter++;
                }
                hostCounter++;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    /**
     * Reads a {@code long} value from a child XML element, returning
     * {@code defaultVal} if the element is absent or cannot be parsed.
     */
    private static long parseLongTag(Element parent, String tagName, long defaultVal) {
        NodeList nl = parent.getElementsByTagName(tagName);
        if (nl.getLength() > 0 && nl.item(0) != null) {
            String text = nl.item(0).getTextContent().trim();
            try { return Long.parseLong(text); }
            catch (NumberFormatException ignored) { /* fall through */ }
        }
        return defaultVal;
    }
}
