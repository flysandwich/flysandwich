import matplotlib.pyplot as plt
from ns import ns
import pdb
def SayHello():
    print("👋 Hello at", ns.Simulator.Now().GetSeconds(), "秒")

ns.cppyy.cppdef("""
// Defining the following function as part of the ns3 namespace
// allows it to be accessed later via ns.printCurrentTimeMakeEvent().
// If the namespace is not specified, it will be accessable via
// ns.cppyy.gbl.printCurrentTimeMakeEvent().
namespace ns3
{
    EventImpl* printCurrentTimeMakeEvent(void (*f)())
    {
        return MakeEvent(f);
    }
}
""")

event = ns.printCurrentTimeMakeEvent(SayHello)
ns.Simulator.Schedule(ns.Seconds(3.0), event)
ns.Simulator.Run()
ns.Simulator.Stop()
ns.Simulator.ScheduleNow(event)
ns.Simulator.Destroy()