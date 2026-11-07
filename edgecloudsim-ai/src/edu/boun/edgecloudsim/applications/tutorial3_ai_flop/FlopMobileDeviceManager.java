/*
 * Title:        EdgeCloudSim - tutorial3_ai_flop - FlopMobileDeviceManager
 *
 * Description:
 * Bản copy của tutorial3.SampleMobileDeviceManager (giữ NGUYÊN luồng mạng
 * WLAN/WAN/MAN), CHỈ thêm một việc: khi tạo Task, nạp GPU MFLOP vào Task.tensors
 * nếu TaskProperty là FlopTaskProperty. Nhờ đó DualChannelScheduler tính được
 * processingTime = max(cpuTime, gpuTime).
 *
 * Licence:      GPL - http://www.gnu.org/copyleft/gpl.html
 * (Dựa trên SampleMobileDeviceManager - Copyright (c) 2017, Bogazici University)
 */
package edu.boun.edgecloudsim.applications.tutorial3_ai_flop;

import org.cloudbus.cloudsim.UtilizationModel;
import org.cloudbus.cloudsim.UtilizationModelFull;
import org.cloudbus.cloudsim.Vm;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;

import edu.boun.edgecloudsim.applications.tutorial3.SampleNetworkModel;
import edu.boun.edgecloudsim.core.SimManager;
import edu.boun.edgecloudsim.core.SimSettings;
import edu.boun.edgecloudsim.core.SimSettings.NETWORK_DELAY_TYPES;
import edu.boun.edgecloudsim.edge_client.CpuUtilizationModel_Custom;
import edu.boun.edgecloudsim.edge_client.MobileDeviceManager;
import edu.boun.edgecloudsim.edge_client.Task;
import edu.boun.edgecloudsim.edge_server.EdgeHost;
import edu.boun.edgecloudsim.edge_server.EdgeVM;
import edu.boun.edgecloudsim.network.NetworkModel;
import edu.boun.edgecloudsim.utils.Location;
import edu.boun.edgecloudsim.utils.SimLogger;
import edu.boun.edgecloudsim.utils.TaskProperty;

public class FlopMobileDeviceManager extends MobileDeviceManager {
	private static final int BASE = 100000;

	private static final int UPDATE_MM1_QUEUE_MODEL = BASE + 1;
	private static final int REQUEST_RECEIVED_BY_CLOUD = BASE + 2;
	private static final int REQUEST_RECEIVED_BY_EDGE_DEVICE = BASE + 3;
	private static final int REQUEST_RECEIVED_BY_REMOTE_EDGE_DEVICE = BASE + 4;
	private static final int REQUEST_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_NEIGHBOR = BASE + 5;
	private static final int RESPONSE_RECEIVED_BY_MOBILE_DEVICE = BASE + 6;
	private static final int RESPONSE_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_MOBILE_DEVICE = BASE + 7;

	private static final double MM1_QUEUE_MODEL_UPDATE_INTEVAL = 5;

	private int taskIdCounter=0;

	public FlopMobileDeviceManager() throws Exception{
	}

	@Override
	public void initialize() {
	}

	@Override
	public UtilizationModel getCpuUtilizationModel() {
		return new CpuUtilizationModel_Custom();
	}

	@Override
	public void startEntity() {
		super.startEntity();
		schedule(getId(), SimSettings.CLIENT_ACTIVITY_START_TIME +
				MM1_QUEUE_MODEL_UPDATE_INTEVAL, UPDATE_MM1_QUEUE_MODEL);
	}

	@Override
	protected void submitCloudlets() {
		//do nothing!
	}

	@Override
	protected void processCloudletReturn(SimEvent ev) {
		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();
		Task task = (Task) ev.getData();

		SimLogger.getInstance().taskExecuted(task.getCloudletId());

		if(task.getAssociatedDatacenterId() == SimSettings.CLOUD_DATACENTER_ID){
			double WanDelay = networkModel.getDownloadDelay(SimSettings.CLOUD_DATACENTER_ID, task.getMobileDeviceId(), task);
			if(WanDelay > 0)
			{
				Location currentLocation = SimManager.getInstance().getMobilityModel().getLocation(task.getMobileDeviceId(),CloudSim.clock()+WanDelay);
				if(task.getSubmittedLocation().getServingWlanId() == currentLocation.getServingWlanId())
				{
					networkModel.downloadStarted(task.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
					SimLogger.getInstance().setDownloadDelay(task.getCloudletId(), WanDelay, NETWORK_DELAY_TYPES.WAN_DELAY);
					schedule(getId(), WanDelay, RESPONSE_RECEIVED_BY_MOBILE_DEVICE, task);
				}
				else
				{
					SimLogger.getInstance().failedDueToMobility(task.getCloudletId(), CloudSim.clock());
				}
			}
			else
			{
				SimLogger.getInstance().failedDueToBandwidth(task.getCloudletId(), CloudSim.clock(), NETWORK_DELAY_TYPES.WAN_DELAY);
			}
		}
		else{
			int nextEvent = RESPONSE_RECEIVED_BY_MOBILE_DEVICE;
			int nextDeviceForNetworkModel = SimSettings.GENERIC_EDGE_DEVICE_ID;
			NETWORK_DELAY_TYPES delayType = NETWORK_DELAY_TYPES.WLAN_DELAY;
			double delay = networkModel.getDownloadDelay(task.getAssociatedHostId(), task.getMobileDeviceId(), task);

			EdgeHost host = (EdgeHost)(SimManager.
					getInstance().
					getEdgeServerManager().
					getDatacenterList().get(task.getAssociatedHostId()).
					getHostList().get(0));

			if(host.getLocation().getServingWlanId() != task.getSubmittedLocation().getServingWlanId())
			{
				delay = networkModel.getDownloadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID, SimSettings.GENERIC_EDGE_DEVICE_ID, task);
				nextEvent = RESPONSE_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_MOBILE_DEVICE;
				nextDeviceForNetworkModel = SimSettings.GENERIC_EDGE_DEVICE_ID + 1;
				delayType = NETWORK_DELAY_TYPES.MAN_DELAY;
			}

			if(delay > 0)
			{
				Location currentLocation = SimManager.getInstance().getMobilityModel().getLocation(task.getMobileDeviceId(),CloudSim.clock()+delay);
				if(task.getSubmittedLocation().getServingWlanId() == currentLocation.getServingWlanId())
				{
					networkModel.downloadStarted(currentLocation, nextDeviceForNetworkModel);
					SimLogger.getInstance().setDownloadDelay(task.getCloudletId(), delay, delayType);

					schedule(getId(), delay, nextEvent, task);
				}
				else
				{
					SimLogger.getInstance().failedDueToMobility(task.getCloudletId(), CloudSim.clock());
				}
			}
			else
			{
				SimLogger.getInstance().failedDueToBandwidth(task.getCloudletId(), CloudSim.clock(), delayType);
			}
		}
	}

	@Override
	protected void processOtherEvent(SimEvent ev) {
		if (ev == null) {
			SimLogger.printLine(getName() + ".processOtherEvent(): " + "Error - an event is null! Terminating simulation...");
			System.exit(0);
			return;
		}

		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();

		switch (ev.getTag()) {
			case UPDATE_MM1_QUEUE_MODEL:
			{
				((SampleNetworkModel)networkModel).updateMM1QueeuModel();
				schedule(getId(), MM1_QUEUE_MODEL_UPDATE_INTEVAL, UPDATE_MM1_QUEUE_MODEL);

				break;
			}
			case REQUEST_RECEIVED_BY_CLOUD:
			{
				Task task = (Task) ev.getData();
				networkModel.uploadFinished(task.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
				submitTaskToVm(task, SimSettings.VM_TYPES.CLOUD_VM);
				break;
			}
			case REQUEST_RECEIVED_BY_EDGE_DEVICE:
			{
				Task task = (Task) ev.getData();
				networkModel.uploadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
				submitTaskToVm(task, SimSettings.VM_TYPES.EDGE_VM);
				break;
			}
			case REQUEST_RECEIVED_BY_REMOTE_EDGE_DEVICE:
			{
				Task task = (Task) ev.getData();
				networkModel.uploadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID+1);
				submitTaskToVm(task, SimSettings.VM_TYPES.EDGE_VM);

				break;
			}
			case REQUEST_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_NEIGHBOR:
			{
				Task task = (Task) ev.getData();
				networkModel.uploadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);

				double manDelay =  networkModel.getUploadDelay(SimSettings.GENERIC_EDGE_DEVICE_ID, SimSettings.GENERIC_EDGE_DEVICE_ID, task);
				if(manDelay>0){
					networkModel.uploadStarted(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID+1);
					SimLogger.getInstance().setUploadDelay(task.getCloudletId(), manDelay, NETWORK_DELAY_TYPES.MAN_DELAY);
					schedule(getId(), manDelay, REQUEST_RECEIVED_BY_REMOTE_EDGE_DEVICE, task);
				}
				else
				{
					SimLogger.getInstance().rejectedDueToBandwidth(
							task.getCloudletId(),
							CloudSim.clock(),
							SimSettings.VM_TYPES.EDGE_VM.ordinal(),
							NETWORK_DELAY_TYPES.MAN_DELAY);
				}

				break;
			}
			case RESPONSE_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_MOBILE_DEVICE:
			{
				Task task = (Task) ev.getData();
				networkModel.downloadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID+1);

				double delay = networkModel.getDownloadDelay(task.getAssociatedHostId(), task.getMobileDeviceId(), task);

				if(delay > 0)
				{
					Location currentLocation = SimManager.getInstance().getMobilityModel().getLocation(task.getMobileDeviceId(),CloudSim.clock()+delay);
					if(task.getSubmittedLocation().getServingWlanId() == currentLocation.getServingWlanId())
					{
						networkModel.downloadStarted(currentLocation, SimSettings.GENERIC_EDGE_DEVICE_ID);
						SimLogger.getInstance().setDownloadDelay(task.getCloudletId(), delay, NETWORK_DELAY_TYPES.WLAN_DELAY);
						schedule(getId(), delay, RESPONSE_RECEIVED_BY_MOBILE_DEVICE, task);
					}
					else
					{
						SimLogger.getInstance().failedDueToMobility(task.getCloudletId(), CloudSim.clock());
					}
				}
				else
				{
					SimLogger.getInstance().failedDueToBandwidth(task.getCloudletId(), CloudSim.clock(), NETWORK_DELAY_TYPES.WLAN_DELAY);
				}

				break;
			}
			case RESPONSE_RECEIVED_BY_MOBILE_DEVICE:
			{
				Task task = (Task) ev.getData();

				if(task.getAssociatedDatacenterId() == SimSettings.CLOUD_DATACENTER_ID) {
					networkModel.downloadFinished(task.getSubmittedLocation(), SimSettings.CLOUD_DATACENTER_ID);
				} else {
					networkModel.downloadFinished(task.getSubmittedLocation(), SimSettings.GENERIC_EDGE_DEVICE_ID);
				}

				SimLogger.getInstance().taskEnded(task.getCloudletId(), CloudSim.clock());
				break;
			}
			default:
				SimLogger.printLine(getName() + ".processOtherEvent(): " + "Error - event unknown by this DatacenterBroker. Terminating simulation...");
				System.exit(0);
				break;
		}
	}

	@Override
	public void submitTask(TaskProperty edgeTask) {
		int vmType=0;
		int nextEvent=0;
		int nextDeviceForNetworkModel;
		NETWORK_DELAY_TYPES delayType;
		double delay=0;

		NetworkModel networkModel = SimManager.getInstance().getNetworkModel();

		//create a task
		Task task = createTask(edgeTask);

		Location currentLocation = SimManager.getInstance().getMobilityModel().
				getLocation(task.getMobileDeviceId(), CloudSim.clock());

		task.setSubmittedLocation(currentLocation);

		SimLogger.getInstance().addLog(task.getMobileDeviceId(),
				task.getCloudletId(),
				task.getTaskType(),
				(int)task.getCloudletLength(),
				(int)task.getCloudletFileSize(),
				(int)task.getCloudletOutputSize());

		long startTime = System.nanoTime();
		int nextHopId = SimManager.getInstance().getEdgeOrchestrator().getDeviceToOffload(task);
		long estimatedTime = System.nanoTime() - startTime;

		SimLogger.getInstance().setOrchestratorOverhead(task.getCloudletId(), estimatedTime);

		if(nextHopId == SimSettings.CLOUD_DATACENTER_ID){
			delay = networkModel.getUploadDelay(task.getMobileDeviceId(), SimSettings.CLOUD_DATACENTER_ID, task);
			vmType = SimSettings.VM_TYPES.CLOUD_VM.ordinal();
			nextEvent = REQUEST_RECEIVED_BY_CLOUD;
			delayType = NETWORK_DELAY_TYPES.WAN_DELAY;
			nextDeviceForNetworkModel = SimSettings.CLOUD_DATACENTER_ID;
		}
		else {
			delay = networkModel.getUploadDelay(task.getMobileDeviceId(), SimSettings.GENERIC_EDGE_DEVICE_ID, task);
			vmType = SimSettings.VM_TYPES.EDGE_VM.ordinal();
			nextEvent = REQUEST_RECEIVED_BY_EDGE_DEVICE;
			delayType = NETWORK_DELAY_TYPES.WLAN_DELAY;
			nextDeviceForNetworkModel = SimSettings.GENERIC_EDGE_DEVICE_ID;
		}

		if(delay>0){

			Vm selectedVM = SimManager.getInstance().getEdgeOrchestrator().getVmToOffload(task, nextHopId);

			if(selectedVM != null){
				task.setAssociatedDatacenterId(nextHopId);
				task.setAssociatedHostId(selectedVM.getHost().getId());
				task.setAssociatedVmId(selectedVM.getId());

				getCloudletList().add(task);
				bindCloudletToVm(task.getCloudletId(), selectedVM.getId());

				if(selectedVM instanceof EdgeVM){
					EdgeHost host = (EdgeHost)(selectedVM.getHost());

					if(host.getLocation().getServingWlanId() != task.getSubmittedLocation().getServingWlanId()){
						nextEvent = REQUEST_RECEIVED_BY_EDGE_DEVICE_TO_RELAY_NEIGHBOR;
					}
				}
				networkModel.uploadStarted(currentLocation, nextDeviceForNetworkModel);

				SimLogger.getInstance().taskStarted(task.getCloudletId(), CloudSim.clock());
				SimLogger.getInstance().setUploadDelay(task.getCloudletId(), delay, delayType);

				schedule(getId(), delay, nextEvent, task);
			}
			else{
				SimLogger.getInstance().rejectedDueToVMCapacity(task.getCloudletId(), CloudSim.clock(), vmType);
			}
		}
		else
		{
			SimLogger.getInstance().rejectedDueToBandwidth(task.getCloudletId(), CloudSim.clock(), vmType, delayType);
		}
	}

	private void submitTaskToVm(Task task, SimSettings.VM_TYPES vmType) {
		schedule(getVmsToDatacentersMap().get(task.getVmId()), 0, CloudSimTags.CLOUDLET_SUBMIT, task);

		SimLogger.getInstance().taskAssigned(task.getCloudletId(),
				task.getAssociatedDatacenterId(),
				task.getAssociatedHostId(),
				task.getAssociatedVmId(),
				vmType.ordinal());
	}

	private Task createTask(TaskProperty edgeTask){
		UtilizationModel utilizationModel = new UtilizationModelFull();
		UtilizationModel utilizationModelCPU = getCpuUtilizationModel();

		Task task = new Task(edgeTask.getMobileDeviceId(), ++taskIdCounter,
				edgeTask.getLength(), edgeTask.getPesNumber(),
				edgeTask.getInputFileSize(), edgeTask.getOutputFileSize(),
				utilizationModelCPU, utilizationModel, utilizationModel);

		task.setUserId(this.getId());
		task.setTaskType(edgeTask.getTaskType());

		// *** TÁCH CPU/GPU: nạp GPU MFLOP vào Task.tensors (kênh GPU) ***
		// cloudletLength = CPU MI (kênh CPU); DualChannelScheduler dùng cả hai.
		// vramRequired (0 nếu XML không có <gpu_vram_mb>, tương thích ngược) --
		// dùng bởi DRLEdgeOrchestrator cho Constraint C5; app này (SampleEdgeOrchestrator)
		// không đọc trường này nên hành vi không đổi.
		if (edgeTask instanceof FlopTaskProperty) {
			task.setTensors(((FlopTaskProperty) edgeTask).getGpuMFLOP());
			task.setVramRequired(((FlopTaskProperty) edgeTask).getVramMB());
		}

		if (utilizationModelCPU instanceof CpuUtilizationModel_Custom) {
			((CpuUtilizationModel_Custom)utilizationModelCPU).setTask(task);
		}

		return task;
	}
}
