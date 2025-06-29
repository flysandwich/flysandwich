from ns import ns
#ns.LogComponentEnable("TrafficControlHelper", ns.LOG_LEVEL_INFO)
#ns.LogComponentEnable("Rx", ns.LOG_LEVEL_INFO)
#ns.LogComponentEnable("PacketSink", ns.LOG_LEVEL_INFO)
#ns.LogComponentEnable("OnOffApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("SpectrumWifiPhy", ns.LOG_LEVEL_INFO)
#ns.LogComponentEnable("WifiPhy", ns.LOG_LEVEL_INFO)
ns.cppyy.cppdef("""
    #ifndef pymakeevent
    #define pymakeevent
    namespace ns3 {

    void CheckAndStartAppLoop(Ptr<Node> node, ApplicationContainer apps, Vector target, double radius, Time interval) {
        Ptr<MobilityModel> mobility = node->GetObject<MobilityModel>();
        if (!mobility) {
            std::cout << "[ERROR] 無法取得 MobilityModel" << std::endl;
            return;
        }

        Vector pos = mobility->GetPosition();
        double dx = pos.x - target.x;
        double dy = pos.y - target.y;
        double dz = pos.z - target.z;
        double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        std::cout << "[" << Simulator::Now().GetSeconds() << "s] 距離：" << dist << std::endl;

        if (dist <= radius) {
            //apps.Start(Seconds(Simulator::Now().GetSeconds()));
            std::cout << "[✓] 已啟動 App 傳輸" << std::endl;
        } else {
            // 尚未進入範圍，排下一次
            Simulator::Schedule(interval, &CheckAndStartAppLoop, node, apps, target, radius, interval);
        }
    }

    }
#endif
""")

ns.cppyy.cppdef("""
   #ifndef pymakeeventl
   #define pymakeeventl
   namespace ns3
   {
       EventImpl* pythonMakeEvent(void (*f)())
       {
           return MakeEvent(f);
       }
   }
   #endif
""")


aps = ns.NodeContainer()
aps.Create(1)
# Create a topology with 10 stations/user devices
stas = ns.NodeContainer()
stas.Create(1)
mobilityHelper = ns.MobilityHelper()
# Set mobility of the access point
centralPosition = ns.CreateObject[ns.ListPositionAllocator]()
centralPosition.__deref__().Add(ns.Vector(0, 0, 0))
mobilityHelper.SetMobilityModel("ns3::ConstantPositionMobilityModel")
mobilityHelper.SetPositionAllocator(centralPosition)
mobilityHelper.Install(aps)
# Set mobility of user devices
uePositions = ns.CreateObject[ns.ListPositionAllocator]()
uePositions.__deref__().Add(ns.Vector(1, 0, 0))
mobilityHelper.SetMobilityModel("ns3::ConstantVelocityMobilityModel")
mobilityHelper.SetPositionAllocator(uePositions)
mobilityHelper.Install(stas)
for sta in range(stas.GetN()):
    stas.Get(sta).GetObject[ns.MobilityModel]().__deref__().SetVelocity(ns.Vector3D(-0.2,0,0))
# Configure wifi channel
spectrumChannel = ns.CreateObject[ns.MultiModelSpectrumChannel]()
lossModel = ns.CreateObject[ns.LogDistancePropagationLossModel]()
spectrumChannel.__deref__().AddPropagationLossModel(lossModel)
# Configure the wifi stack
wifiHelper = ns.WifiHelper()
wifiHelper.SetStandard(ns.WIFI_STANDARD_80211ac)  # using WiFi 6
wifiPhyHelper = ns.SpectrumWifiPhyHelper()
wifiPhyHelper.Set("ChannelSettings",
                  ns.StringValue("{50, 160, BAND_5GHZ, 0}")
                  )  # 160MHz-wide band in the 5GHz range
wifiPhyHelper.SetChannel(spectrumChannel)
# Create a new wifi ssid and connect the AP and STAs to it
wifiSsid = ns.Ssid("ns3-80211ac")
wifiMacHelper = ns.WifiMacHelper()
wifiMacHelper.SetType("ns3::StaWifiMac", "Ssid", ns.SsidValue(wifiSsid))
stasDevices = wifiHelper.Install(wifiPhyHelper, wifiMacHelper, stas)
staswifiNetDevice = stasDevices.Get(0).__deref__().GetObject["WifiNetDevice"]()
staswifiphy = staswifiNetDevice.GetPhys()[0]
# Install the AP interface
wifiMacHelper.SetType("ns3::ApWifiMac", "Ssid", ns.SsidValue(wifiSsid))
apsDevices = wifiHelper.Install(wifiPhyHelper, wifiMacHelper, aps)
apswifiNetDevice = apsDevices.Get(0).__deref__().GetObject["WifiNetDevice"]()
apswifiphy = apswifiNetDevice.GetPhys()[0]
# Setup IP stack and addresses
stack = ns.InternetStackHelper()
stack.Install(aps)
stack.Install(stas)
address = ns.Ipv4AddressHelper()
address.SetBase("192.168.1.0", "255.255.255.0")
apsNodeInterfaces = address.Assign(apsDevices)
staNodeInterfaces = address.Assign(stasDevices)
# Schedule getNodeCoordinates to run after 1 second of simulation
allNodes = ns.NodeContainer()
allNodes.Add(stas)
allNodes.Add(aps)
# Setup applications
port = 9
localAddress = ns.InetSocketAddress(ns.Ipv4Address.GetAny(), port).ConvertTo()
server = ns.PacketSinkHelper("ns3::TcpSocketFactory", localAddress)
serverApp = server.Install(aps)
serverApp.Start(ns.Seconds(0))
serverApp.Stop(ns.Seconds(20))

client = ns.OnOffHelper("ns3::TcpSocketFactory", staNodeInterfaces.GetAddress(0).ConvertTo())
client.SetAttribute("OnTime",
                    ns.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
client.SetAttribute("OffTime",
                    ns.StringValue("ns3::ConstantRandomVariable[Constant=1]"))
client.SetAttribute("PacketSize", ns.UintegerValue(1400))
client.SetAttribute("DataRate", ns.DataRateValue(1500000))
remoteAddress = ns.InetSocketAddress(apsNodeInterfaces.GetAddress(0), port).ConvertTo()
client.SetAttribute("Remote", ns.AddressValue(remoteAddress))
clientApp = client.Install(stas.Get(0))
clientApp.Start(ns.Seconds(0))
clientApp.Stop(ns.Seconds(20))
# === 模擬節點在 2 秒後移動到目標位置 ===
#def move_node():
#    print("⚙️ 節點移動到目標位置")
#    mob_model.SetPosition(ns.Vector(50.0, 0.0, 0.0))
#
#ns.Simulator.Schedule(ns.Seconds(2.0), move_node)

# Collect flow statistics
flowHelper = ns.FlowMonitorHelper()
flowMonitor = flowHelper.InstallAll()
ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()
def ServerWifiStart():
    print("Server On 被呼叫時間：", ns.Simulator.Now().GetSeconds())
    apswifiphy.ResumeFromOff()

def ServerWifiStop():
    print("Server Off 被呼叫時間：", ns.Simulator.Now().GetSeconds())
    apswifiphy.SetOffMode()

def ClientWifiStart():
    print("Client On 被呼叫時間：", ns.Simulator.Now().GetSeconds())
    staswifiphy.ResumeFromOff()
    #ipv4, index = staNodeInterfaces.Get(0)
    #interface = ipv4.GetInterface(index)
    #interface.SetUp()

def ClientWifiStop():
    print("Client Off 被呼叫時間：", ns.Simulator.Now().GetSeconds())
    staswifiphy.SetOffMode()
    #ipv4, index = staNodeInterfaces.Get(0)
    #interface = ipv4.GetInterface(index)
    #interface.SetDown()

client_event_start = ns.pythonMakeEvent(ClientWifiStart)
client_event_stop = ns.pythonMakeEvent(ClientWifiStop)
server_event_start = ns.pythonMakeEvent(ServerWifiStart)
server_event_stop = ns.pythonMakeEvent(ServerWifiStop)
ns.Simulator.Schedule(ns.Seconds(6.5), client_event_stop)
ns.Simulator.Schedule(ns.Seconds(13), client_event_start)
# === 呼叫檢查函數 ===
target = aps.Get(0).GetObject[ns.MobilityModel]().__deref__().GetPosition()
radius = 0.1
interval = ns.Seconds(0.2)

#ns.CheckAndStartAppLoop(stas.Get(0), clientApp, target, radius, interval)
#result = ns.CheckAndStartApp(node, apps, target, radius)
#print(f"🔍 初始檢查結果：{result}")

# === 執行模擬 ===

ns.Simulator.Stop(ns.Seconds(15.0))
ns.Simulator.Run()
flowMonitor.CheckForLostPackets()
classifier = flowHelper.GetClassifier()
lost_packets = []
for flow_id, flow_stats in flowMonitor.GetFlowStats():
    t = classifier.FindFlow(flow_id)
    if flow_id >= 4:
        break
    proto = {6: 'TCP', 17: 'UDP'}[t.protocol]

    print("FlowID: %i (%s %s/%s --> %s/%i)" % \
          (flow_id, proto, t.sourceAddress, t.sourcePort, t.destinationAddress, t.destinationPort))
    print("  Lost Packets: ", flow_stats.lostPackets)
    print("  Tx Packets:", flow_stats.txBytes)
    print("  Rx Packets:", flow_stats.rxBytes)
lost_packets.append((flow_stats.txPackets-flow_stats.rxPackets)*100/flow_stats.txPackets)
ns.Simulator.Destroy()