/*
 * Title:        EdgeCloudSim - DualChannelCloudServerManager  (Phase 2)
 *
 * Description:
 * Creates cloud VMs equipped with DualChannelScheduler so that cloud-side
 * AI inference also uses processingTime = max(cpuTime, gpuTime).
 *
 * Cloud configuration:
 *   VM.mips  = mips_for_cloud_vm from config.properties  (GPU MFLOPS of T4 = 3,500,000)
 *   cpuMips  = CLOUD_CPU_MIPS hardcoded below             (Intel Xeon ~100,000 MIPS)
 *
 * Cloud T4 example (ResNet-50 batch=1):
 *   cpuTime = 24 MI   / 100,000     =  0.24 ms
 *   gpuTime = 10,178  / 3,500,000   =  2.9  ms
 *   effectiveTime = max(0.24, 2.9)  =  2.9  ms  ✓  (cloud GPU inference faster than edge)
 *
 * Note: DualChannelScheduler reads vmMips from getCurrentMipsShare() at submit
 * time, so it automatically picks up the correct cloud GPU MFLOPS at runtime.
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 */
package edu.boun.edgecloudsim.gpu_dual;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.Datacenter;
import org.cloudbus.cloudsim.DatacenterCharacteristics;
import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.VmAllocationPolicy;
import org.cloudbus.cloudsim.VmSchedulerSpaceShared;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import edu.boun.edgecloudsim.cloud_server.CloudServerManager;
import edu.boun.edgecloudsim.cloud_server.CloudVM;
import edu.boun.edgecloudsim.cloud_server.CloudVmAllocationPolicy_Custom;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;

/**
 * Phase 2 cloud server manager.
 *
 * <p>Same structure as {@link edu.boun.edgecloudsim.cloud_server.DefaultCloudServerManager}
 * but creates {@link CloudVM} instances with {@link DualChannelScheduler}
 * instead of {@code CloudletSchedulerTimeShared}.</p>
 *
 * <p>VM.mips stays at {@code mips_for_cloud_vm} (T4 GPU MFLOPS = 3,500,000).
 * DualChannelScheduler uses {@link #CLOUD_CPU_MIPS} as the CPU channel divisor.</p>
 */
public class DualChannelCloudServerManager extends CloudServerManager {

    /**
     * Intel Xeon Platinum approximate MIPS for a single core used in cloud VMs.
     * Used as cpuMips in DualChannelScheduler for cloud-side preprocessing.
     * Preprocessing (resize/normalise) is negligible vs GPU inference at this speed:
     *   cpuTime = 24 MI / 100,000 = 0.24 ms  vs  gpuTime ~3 ms  → GPU dominates ✓
     */
    public static final long CLOUD_CPU_MIPS = 100_000L;

    @Override public void initialize() {}

    @Override
    public VmAllocationPolicy getVmAllocationPolicy(List<? extends Host> hostList,
                                                     int dataCenterIndex) {
        return new CloudVmAllocationPolicy_Custom(hostList, dataCenterIndex);
    }

    @Override
    public void startDatacenters() throws Exception {
        localDatacenter = createDatacenter(SimSettings.CLOUD_DATACENTER_ID);
    }

    @Override
    public void terminateDatacenters() {
        localDatacenter.shutdownEntity();
    }

    @Override
    public void createVmList(int brokerId) {
        // Cloud VM IDs start after edge VM IDs (same as DefaultCloudServerManager)
        int vmCounter = SimSettings.getInstance().getNumOfEdgeVMs();

        for (int i = 0; i < SimSettings.getInstance().getNumOfCloudHost(); i++) {
            vmList.add(i, new ArrayList<>());
            for (int j = 0; j < SimSettings.getInstance().getNumOfCloudVMsPerHost(); j++) {
                String vmm     = "Xen";
                int    cores   = SimSettings.getInstance().getCoreForCloudVM();
                double mips    = SimSettings.getInstance().getMipsForCloudVM();
                int    ram     = SimSettings.getInstance().getRamForCloudVM();
                long   storage = SimSettings.getInstance().getStorageForCloudVM();
                long   bw      = 0;

                // Phase 2: cloud VMs use DualChannelSchedulerTimeShared (NOT SpaceShared).
                // TimeShared mirrors DefaultCloudServerManager and prevents O(n) queue
                // buildup when many tasks are routed to cloud under NETWORK_BASED policy.
                CloudVM vm = new CloudVM(vmCounter, brokerId, mips, cores, ram, bw, storage,
                        vmm, new DualChannelSchedulerTimeShared(CLOUD_CPU_MIPS));
                vmList.get(i).add(vm);
                vmCounter++;
            }
        }
    }

    @Override
    public double getAvgUtilization() {
        double total = 0;
        double count = 0;
        List<? extends Host> list = localDatacenter.getHostList();
        for (int h = 0; h < list.size(); h++) {
            List<CloudVM> vms = SimManager.getInstance().getCloudServerManager().getVmList(h);
            for (CloudVM vm : vms) {
                total += vm.getCloudletScheduler()
                           .getTotalUtilizationOfCpu(org.cloudbus.cloudsim.core.CloudSim.clock());
                count++;
            }
        }
        return count > 0 ? total / count : 0;
    }

    // ------------------------------------------------------------------
    // Datacenter creation — identical to DefaultCloudServerManager
    // ------------------------------------------------------------------

    private Datacenter createDatacenter(int index) throws Exception {
        String arch = "x86";
        String os   = "Linux";
        String vmm  = "Xen";
        double costPerBw = 0, costPerSec = 0, costPerMem = 0, costPerStorage = 0;

        List<Host> hostList = createHosts();

        String name = "CloudDatacenter_" + index;
        double timeZone = 3.0;
        LinkedList<Storage> storageList = new LinkedList<>();

        DatacenterCharacteristics chars = new DatacenterCharacteristics(
                arch, os, vmm, hostList, timeZone,
                costPerSec, costPerMem, costPerStorage, costPerBw);

        return new Datacenter(name, chars,
                getVmAllocationPolicy(hostList, index),
                storageList, 0);
    }

    private List<Host> createHosts() {
        List<Host> hostList = new ArrayList<>();
        for (int i = 0; i < SimSettings.getInstance().getNumOfCloudHost(); i++) {
            // Mirrors DefaultCloudServerManager.createHosts() exactly:
            //   numOfCores  = coresPerVm  * numOfVMPerHost
            //   mips per PE = mipsPerVm   * numOfVMPerHost  (VmSchedulerSpaceShared peCapacity check)
            //   ram         = ramPerVm    * numOfVMPerHost
            //   storage     = storagePerVm * numOfVMPerHost
            //   hostId      = i + numOfEdgeHosts            (avoid conflict with edge host IDs)
            int    numOfVMPerHost = SimSettings.getInstance().getNumOfCloudVMsPerHost();
            int    numOfCores = SimSettings.getInstance().getCoreForCloudVM() * numOfVMPerHost;
            double mips       = SimSettings.getInstance().getMipsForCloudVM() * numOfVMPerHost;
            int    ram        = SimSettings.getInstance().getRamForCloudVM()  * numOfVMPerHost;
            long   storage    = SimSettings.getInstance().getStorageForCloudVM() * numOfVMPerHost;
            long   bw         = 0; // cloud bandwidth managed by WAN model, not host provisioner

            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < numOfCores; j++) {
                peList.add(new Pe(j, new PeProvisionerSimple(mips)));
            }

            hostList.add(new Host(
                    i + SimSettings.getInstance().getNumOfEdgeHosts(), // unique ID after edge hosts
                    new RamProvisionerSimple(ram),
                    new BwProvisionerSimple(bw),
                    storage,
                    peList,
                    new VmSchedulerSpaceShared(peList)));
        }
        return hostList;
    }
}
