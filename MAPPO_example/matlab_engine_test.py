import matlab.engine

eng = matlab.engine.start_matlab()
carrier = eng.nrCarrierConfig('NCellID', 42,'NSlot',0)
NSizeBWP = 25
pdsch = eng.nrPDSCHConfig('Modulation','16QAM','RNTI',1005,'NSizeBWP',25,'NStartBWP',10,'PRBSet',eng.int16(eng.linspace(0.0, 24.0,25.0)))
#pdsch[] = 
#pdsch[] = 
#pdsch['NID']= 
#pdsch['NSizeBWP'] = 
#pdsch['NStartBWP'] = 
#pdsch['PRBSet'] = eng.eval("0:pdsch.NSizeBWP-1", nargout=1)

ind,info = eng.nrPDSCHIndices(carrier,pdsch,'IndexStyle','subscript','IndexOrientation','bwp',nargout = 2)
print(info)