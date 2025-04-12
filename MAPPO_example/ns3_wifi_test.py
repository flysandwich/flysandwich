from ns import ns
ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)

#Create Wi-Fi node and print ID
wifiVehicleNode = ns.NodeContainer()
wifiVehicleNode.Create(n = 1)
for i in range(wifiVehicleNode.GetN()):
    node = wifiVehicleNode.Get(i)
    print(f"Node {i} ID: {node.GetId()}")
wifiUavNode = ns.NodeContainer()
wifiUavNode.Create(n = 5)
for i in range(wifiUavNode.GetN()):
    node = wifiUavNode.Get(i)
    print(f"Node {i} ID: {node.GetId()}")
#Create a channel helper and phy helper, and then create the channel
channel = ns.YansWifiChannelHelper.Default()
phy = ns.YansWifiPhyHelper()
phy.SetChannel(channel.Create())

#Create a WifiMacHelper, which is reused across mobile vehicle and UAV configurations
mac = ns.WifiMacHelper()
packetSocket = ns.PacketSocketHelper()
packetSocket.Install(wifiVehicleNode)
packetSocket.Install(wifiUavNode)

#Create a WifiHelper, which will use the above helpers to create and install Wifi devices.  Configure a Wifi standard to use, which will align various parameters in the Phy and Mac to standard defaults.
wifi = ns.WifiHelper()
wifi.SetStandard(ns.WIFI_STANDARD_80211ac)

#Declare NetDeviceContainers to hold the container returned by the helper
wifiVehicleDevices = ns.NetDeviceContainer()
wifiUavDevice = ns.NetDeviceContainer() 

#Mobility setter
mobility = ns.MobilityHelper()

#Perform the installation
mac.SetType("ns3::StaWifiMac")
wifiVehicleDevices = wifi.Install(phy, mac, wifiVehicleNode)
mac.SetType("ns3::ApWifiMac")
wifiUavDevice = wifi.Install(phy, mac, wifiUavNode)
mobility.Install(wifiVehicleNode)
mobility.Install(wifiUavNode)


ns.cppyy.cppdef(
    """
    using namespace ns3;
    void AdvancePosition(Ptr<Node> node){
        Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
        Vector pos = mob->GetPosition();
        pos.x += 5.0;
        if (pos.x >= 210.0)
            return;
        mob->SetPosition(pos);
        Simulator::Schedule(Seconds(1), AdvancePosition, node);
    }"""
)
ns.Simulator.Schedule(ns.Seconds(1), ns.cppyy.gbl.AdvancePosition, wifiUavNode.Get(0))

socket = ns.PacketSocketAddress()
socket.SetSingleDevice(wifiVehicleDevices.Get(0).GetIfIndex())
wifiVehicleDevices.Get(1)
socket.SetPhysicalAddress(wifiVehicleDevices.Get(0).GetAddress())
socket.SetProtocol(1)
onoff = ns.OnOffHelper("ns3::PacketSocketFactory", socket.ConvertTo())
onoff.SetConstantRate(ns.DataRate("500kb/s"))
apps = onoff.Install(ns.NodeContainer(wifiVehicleNode.Get(0)))
apps.Start(ns.Seconds(0.5))
apps.Stop(ns.Seconds(43))
ns.Simulator.Stop(ns.Seconds(44))



ns.Simulator.Run()
ns.Simulator.Destroy()